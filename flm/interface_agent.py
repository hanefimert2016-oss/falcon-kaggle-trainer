from __future__ import annotations

import json
from pathlib import Path
import re

import torch
from tokenizers import Tokenizer

from flm.core import FLMCore, Program
from flm.core.semantic_compiler import SemanticCompiler
from flm.core.renderer import DeterministicRenderer
from flm.models.core_memory import CoreMemoryBank
from flm.models.interface_transformer import InterfaceConfig, InterfaceTransformer


ROLE_MARKERS={
    "system":"<|system|>",
    "user":"<|user|>",
    "assistant":"<|assistant|>",
    "tool":"<|tool_result|>",
}


def encode_history(tok:Tokenizer,messages:list[dict],append_assistant=True):
    ids=[]
    bos=tok.token_to_id("<bos>")
    end=tok.token_to_id("<|end|>")
    if bos is not None:
        ids.append(bos)
    for msg in messages:
        marker=ROLE_MARKERS.get(str(msg.get("role") or ""))
        if marker is None:
            continue
        mid=tok.token_to_id(marker)
        if mid is None:
            raise RuntimeError(f"missing tokenizer role marker {marker}")
        ids.append(mid)
        ids.extend(tok.encode(str(msg.get("content") or ""),add_special_tokens=False).ids)
        if msg.get("role")=="assistant" and end is not None:
            ids.append(end)
    if append_assistant:
        aid=tok.token_to_id("<|assistant|>")
        if aid is None:
            raise RuntimeError("missing <|assistant|>")
        ids.append(aid)
    return ids


class InterfaceAgent:
    """Language/code interface around the training-free FLM Core."""

    def __init__(
        self,
        checkpoint:str|Path,
        tokenizer:str|Path|None=None,
        *,
        device:str|None=None,
        core_memory:str|Path|None=None,
        temperature:float=0.2,
        top_k:int=20,
        max_new:int=768,
        render_temperature:float=0.75,
        render_top_k:int=50,
        render_top_p:float=0.92,
        repetition_penalty:float=1.08,
    ):
        checkpoint=Path(checkpoint)
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if ck.get("format")!="flm-semantic-v07-interface":
            raise RuntimeError(f"wrong interface checkpoint format: {ck.get('format')!r}")
        cfg=InterfaceConfig(**ck["config"])
        model=InterfaceTransformer(cfg)
        missing,unexpected=model.load_state_dict(ck["model"],strict=True)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
        if device is None:
            device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=torch.device(device)
        self.model=model.to(self.device).eval()
        self.cfg=cfg
        tokenizer=Path(tokenizer) if tokenizer else checkpoint.parent/"tokenizer.json"
        self.tokenizer=Tokenizer.from_file(str(tokenizer))
        self.temperature=float(temperature)
        self.top_k=int(top_k)
        self.max_new=int(max_new)
        self.render_temperature=float(render_temperature)
        self.render_top_k=int(render_top_k)
        self.render_top_p=float(render_top_p)
        self.repetition_penalty=float(repetition_penalty)
        self._recent_answers: list[str] = []
        self.semantic_compiler=SemanticCompiler()
        self.deterministic_renderer=DeterministicRenderer()

        memory_path=Path(core_memory) if core_memory else checkpoint.parent/"core_memory.pt"
        if memory_path.is_file():
            self.model.attach_core_memory(CoreMemoryBank.load(memory_path).to(self.device))

    @torch.inference_mode()
    def generate(
        self,
        messages:list[dict],
        *,
        max_new:int|None=None,
        temperature:float|None=None,
        top_k:int|None=None,
        top_p:float=1.0,
        repetition_penalty:float=1.0,
    )->str:
        ids=encode_history(self.tokenizer,messages)
        eos_ids={
            self.tokenizer.token_to_id("<|end|>"),
            self.tokenizer.token_to_id("<eos>"),
        }
        eos_ids.discard(None)
        generated=[]
        for _ in range(int(max_new or self.max_new)):
            context=ids[-self.cfg.seq_len:]
            x=torch.tensor([context],dtype=torch.long,device=self.device)
            logits,_=self.model(x)
            z=logits[0,-1].float()
            temp=self.temperature if temperature is None else float(temperature)
            k_cfg=self.top_k if top_k is None else int(top_k)
            if repetition_penalty>1.0 and generated:
                seen=torch.tensor(sorted(set(generated)),device=z.device,dtype=torch.long)
                z[seen]=torch.where(z[seen]>=0,z[seen]/repetition_penalty,z[seen]*repetition_penalty)
            if temp<=0:
                nxt=int(z.argmax())
            else:
                z=z/max(temp,1e-5)
                k=min(max(1,k_cfg),z.numel())
                values,indices=torch.topk(z,k)
                probs=torch.softmax(values,-1)
                if top_p<1.0:
                    sorted_probs,order=torch.sort(probs,descending=True)
                    cumulative=torch.cumsum(sorted_probs,dim=-1)
                    keep=cumulative<=max(1e-4,float(top_p))
                    keep[0]=True
                    filtered=torch.zeros_like(probs)
                    filtered[order[keep]]=probs[order[keep]]
                    probs=filtered/filtered.sum()
                nxt=int(indices[torch.multinomial(probs,1)])
            if nxt in eos_ids:
                break
            ids.append(nxt); generated.append(nxt)
        return self.tokenizer.decode(generated,skip_special_tokens=False).strip()

    @staticmethod
    def _extract_ir(text:str)->str:
        m=re.search(
            r"<\|semantic_ir\|>(.*?)<\|semantic_end\|>",
            text,re.S,
        )
        if not m:
            raise ValueError("Interface Transformer did not emit Semantic IR")
        return m.group(1).strip()

    def compile(self,prompt:str)->Program:
        # Explicit canonical TR/EN forms do not need the neural interface at all.
        # This keeps a genuine training-free path for facts, rules, queries and
        # strict arithmetic requests.
        try:
            return self.semantic_compiler.compile_any(prompt)
        except ValueError:
            pass

        system=(
            "You are the only trainable FLM Interface Transformer. Translate the "
            "user's language/code request into executable FLM Semantic IR JSON inside "
            "<|semantic_ir|>...<|semantic_end|>. Do not solve logical, code, math, UI "
            "or planning steps yourself. FLM Core executes facts, rules, queries and "
            "operations. Use ARITHMETIC, ANALYZE_CODE, STATE_PLAN, UI_PLAN or VERIFY "
            "when appropriate."
        )
        raw=self.generate([
            {"role":"system","content":system},
            {"role":"user","content":prompt},
        ],temperature=0.0,top_k=1)
        return Program.from_json(self._extract_ir(raw))

    def answer(self,prompt:str,core:FLMCore)->str:
        # Canonical facts, queries, arithmetic and code-analysis requests are
        # answered entirely by FLM Core; no Transformer generation is needed.
        try:
            direct_program=self.semantic_compiler.compile_any(prompt)
        except ValueError:
            direct_program=None
        if direct_program is not None:
            direct_result=core.execute(direct_program)
            return self.deterministic_renderer.render(prompt,direct_result)

        program=self.compile(prompt)
        result=core.execute(program)
        payload={
            "results":[
                {
                    "answer":x.answer,
                    "query":x.query.to_dict(),
                    "proof":x.proof,
                    "rounds":x.rounds,
                    "derived_facts":x.derived_facts,
                }
                for x in result.results
            ],
            "operation_results":result.operation_results,
            "memory_facts":result.memory_facts,
            "memory_rules":result.memory_rules,
        }
        ir="<|semantic_ir|>"+program.to_json()+"<|semantic_end|>"
        core_result="<|core_result|>"+json.dumps(
            payload,ensure_ascii=False,separators=(",",":")
        )+"<|core_end|>"
        render_messages=[
            {
                "role":"system",
                "content":(
                    "Render the verified FLM Core result naturally. Treat the Core payload as "
                    "meaning, not as wording to copy. Do not quote or mirror the user's source "
                    "text unless an exact quote is explicitly requested. Synthesize a fresh, "
                    "concise answer in the user's language, preserve every verified fact, and "
                    "never contradict the Core result. Vary phrasing naturally across runs "
                    "while keeping the same meaning."
                ),
            },
            {"role":"user","content":prompt},
            {"role":"assistant","content":ir},
            {"role":"tool","content":core_result},
        ]
        answer=""
        for _ in range(3):
            candidate=self.generate(
                render_messages,
                max_new=384,
                temperature=self.render_temperature,
                top_k=self.render_top_k,
                top_p=self.render_top_p,
                repetition_penalty=self.repetition_penalty,
            ).strip()
            answer=candidate
            if candidate and candidate not in self._recent_answers[-4:]:
                break
        if answer:
            self._recent_answers.append(answer)
            self._recent_answers=self._recent_answers[-8:]
        return answer
