from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer

from flm.models.text_lm import ByteCausalLM, TextConfig
from flm.models.core_memory import CoreMemoryBank
from flm.tooling.runtime import ToolRegistry, run_tool_loop


ROLE_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool_result|>",
}


def encode_history(tok: Tokenizer, messages: Iterable[dict], *, append_assistant: bool = True) -> list[int]:
    ids: list[int] = []
    bos = tok.token_to_id("<bos>")
    end = tok.token_to_id("<|end|>")
    if bos is not None:
        ids.append(bos)
    for msg in messages:
        role = str(msg.get("role") or "")
        marker = ROLE_MARKERS.get(role)
        if marker is None:
            continue
        marker_id = tok.token_to_id(marker)
        if marker_id is None:
            raise RuntimeError(f"tokenizer missing role token {marker}")
        ids.append(marker_id)
        ids.extend(tok.encode(str(msg.get("content") or ""), add_special_tokens=False).ids)
        # Match v0.7 SFT serialization exactly.
        if role == "assistant" and end is not None:
            ids.append(end)
    if append_assistant:
        aid = tok.token_to_id("<|assistant|>")
        if aid is None:
            raise RuntimeError("tokenizer missing <|assistant|>")
        ids.append(aid)
    return ids


class V07TextAgent:
    """Load a v0.7 Main/Coder checkpoint and run chat or tool-call loops."""

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer: str | Path,
        *,
        device: str | None = None,
        temperature: float = 0.25,
        top_k: int = 20,
        max_new: int = 256,
        core_memory: str | Path | None = None,
    ):
        checkpoint = Path(checkpoint)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = TextConfig(**ck["config"])
        model = ByteCausalLM(cfg)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch missing={missing} unexpected={unexpected}"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        if cfg.core_memory:
            memory_path = Path(core_memory) if core_memory is not None else checkpoint.parent / "core_memory.pt"
            if memory_path.is_file():
                bank = CoreMemoryBank.load(memory_path, map_location="cpu").to(self.device)
                self.model.attach_core_memory(bank)
                self.core_memory_path = memory_path
            else:
                self.core_memory_path = None
        else:
            self.core_memory_path = None
        self.cfg = cfg
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.max_new = int(max_new)

        for token in (
            "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>",
            "<|tool_call|>", "<|tool_result|>", "<|tool_end|>",
            "<|plan|>", "<|plan_end|>", "<|final|>", "<|end|>",
        ):
            if self.tokenizer.token_to_id(token) is None:
                raise RuntimeError(f"tokenizer missing required token {token}")

    @torch.inference_mode()
    def generate_history(self, messages: list[dict[str, str]]) -> str:
        ids = encode_history(self.tokenizer, messages, append_assistant=True)
        end_ids = {
            self.tokenizer.token_to_id("<|end|>"),
            self.tokenizer.token_to_id("<eos>"),
        }
        end_ids.discard(None)
        generated: list[int] = []
        for _ in range(self.max_new):
            context = ids[-self.cfg.seq_len :]
            x = torch.tensor([context], dtype=torch.long, device=self.device)
            logits, _ = self.model(x)
            z = logits[0, -1].float()
            if self.temperature <= 0:
                nxt = int(z.argmax().item())
            else:
                z = z / max(self.temperature, 1e-5)
                k = min(max(1, self.top_k), z.numel())
                values, indices = torch.topk(z, k)
                probs = torch.softmax(values, dim=-1)
                nxt = int(indices[torch.multinomial(probs, 1)].item())
            if nxt in end_ids:
                break
            ids.append(nxt)
            generated.append(nxt)
        return self.tokenizer.decode(generated, skip_special_tokens=False).strip()

    def chat(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.generate_history(messages)

    def run_tools(
        self,
        messages: list[dict[str, str]],
        registry: ToolRegistry,
        *,
        max_rounds: int = 8,
    ) -> tuple[str, list[dict[str, str]]]:
        """Generate → parse tool calls → execute allow-listed tools → continue."""
        return run_tool_loop(
            self.generate_history,
            messages,
            registry,
            max_rounds=max_rounds,
        )
