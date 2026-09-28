from __future__ import annotations

import json
from pathlib import Path
import re

import torch
from tokenizers import Tokenizer

from flm.core import FLMCore, Program
from flm.core.semantic_compiler import SemanticCompiler
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
        self.semantic_compiler=SemanticCompiler()

        memory_path=Path(core_memory) if core_memory else checkpoint.parent/"core_memory.pt"
        if memory_path.is_file():
            self.model.attach_core_memory(CoreMemoryBank.load(memory_path).to(self.device))

    @torch.inference_mode()
    def generate(self,messages:list[dict],*,max_new:int|None=None)->str:
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
            if self.temperature<=0:
                nxt=int(z.argmax())
            else:
                z=z/max(self.temperature,1e-5)
                k=min(max(1,self.top_k),z.numel())
                values,indices=torch.topk(z,k)
                probs=torch.softmax(values,-1)
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
        # This keeps a genuine training-free path for facts and queries.
        for compiler in (
            self.semantic_compiler.query_canonical,
            self.semantic_compiler.compile_canonical,
        ):
            try:
                return compiler(prompt)
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
        ])
        return Program.from_json(self._extract_ir(raw))

    def answer(self,prompt:str,core:FLMCore)->str:
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
        return self.generate([
            {
                "role":"system",
                "content":"Render the verified FLM Core result naturally. Never contradict the core result.",
            },
            {"role":"user","content":prompt},
            {"role":"assistant","content":ir},
            {"role":"tool","content":core_result},
        ],max_new=384)
