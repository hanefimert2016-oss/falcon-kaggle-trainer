from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


P1 = 2_147_483_647
P2 = 2_147_483_629
M1 = 1_000_003
M2 = 1_000_033


def fingerprint_tokens(tokens: Iterable[int]) -> int:
    h1 = 17
    h2 = 29
    for value in tokens:
        v = int(value) + 1
        h1 = (h1 * M1 + v) % P1
        h2 = (h2 * M2 + v + 7) % P2
    return h1 * P2 + h2


def _fingerprint_windows(windows: torch.Tensor) -> torch.Tensor:
    # windows: [..., order], integer token ids
    h1 = torch.full(windows.shape[:-1], 17, dtype=torch.int64, device=windows.device)
    h2 = torch.full(windows.shape[:-1], 29, dtype=torch.int64, device=windows.device)
    for i in range(windows.size(-1)):
        v = windows[..., i].to(torch.int64) + 1
        h1 = torch.remainder(h1 * M1 + v, P1)
        h2 = torch.remainder(h2 * M2 + v + 7, P2)
    return h1 * P2 + h2


@dataclass
class CoreMemoryMeta:
    vocab_size: int
    order: int
    slots: int
    top_k: int
    accepted_contexts: int
    skipped_collisions: int
    tokens_scanned: int


class CoreMemoryBank(nn.Module):
    """Non-RAG internal conditional memory.

    The bank stores exact hashed token N-gram fingerprints plus likely continuation
    token IDs. No document is retrieved and no text is appended to the prompt.
    Lookup happens inside the model forward pass in O(1) per token position.
    """

    def __init__(
        self,
        fingerprints: torch.Tensor,
        token_ids: torch.Tensor,
        weights: torch.Tensor,
        *,
        vocab_size: int,
        order: int,
    ):
        super().__init__()
        if fingerprints.ndim != 1:
            raise ValueError("fingerprints must be [slots]")
        if token_ids.ndim != 2 or weights.shape != token_ids.shape:
            raise ValueError("token_ids/weights must be [slots, top_k]")
        if token_ids.size(0) != fingerprints.numel():
            raise ValueError("slot count mismatch")
        if order < 1:
            raise ValueError("order must be >=1")
        self.vocab_size = int(vocab_size)
        self.order = int(order)
        self.slots = int(fingerprints.numel())
        self.top_k = int(token_ids.size(1))
        # The memory is data, not trainable model parameters.
        self.register_buffer("fingerprints", fingerprints.to(torch.int64), persistent=False)
        self.register_buffer("token_ids", token_ids.to(torch.int64), persistent=False)
        self.register_buffer("weights", weights.to(torch.float32), persistent=False)

    def _lookup_windows(self, idx: torch.Tensor):
        if idx.size(1) < self.order:
            return None
        windows = idx.unfold(1, self.order, 1)
        fp = _fingerprint_windows(windows)
        slot1 = torch.remainder(fp, self.slots)
        mixed = torch.bitwise_xor(fp, torch.div(fp, max(1, self.slots), rounding_mode="floor"))
        slot2 = torch.remainder(mixed, self.slots)

        fp1 = self.fingerprints[slot1]
        fp2 = self.fingerprints[slot2]
        hit1 = fp1.eq(fp)
        hit2 = (~hit1) & fp2.eq(fp)
        slots = torch.where(hit1, slot1, slot2)
        valid = hit1 | hit2

        toks = self.token_ids[slots]
        weights = self.weights[slots]
        valid = valid.unsqueeze(-1) & toks.ge(0) & weights.gt(0)
        return toks.clamp_min(0), weights * valid.to(weights.dtype), valid

    def residual(self, idx: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        result = torch.zeros(
            idx.size(0), idx.size(1), embedding.embedding_dim,
            dtype=embedding.weight.dtype, device=idx.device,
        )
        found = self._lookup_windows(idx)
        if found is None:
            return result
        toks, weights, valid = found
        emb = embedding(toks)
        denom = weights.sum(-1, keepdim=True).clamp_min(1e-6)
        mem = (emb * weights.unsqueeze(-1)).sum(-2) / denom
        mem = mem * valid.any(-1, keepdim=True).to(mem.dtype)
        result[:, self.order - 1 :, :] = mem
        return result

    def apply_logits(self, idx: torch.Tensor, logits: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 0:
            return logits
        found = self._lookup_windows(idx)
        if found is None:
            return logits
        toks, weights, valid = found
        out = logits.clone()
        tail = out[:, self.order - 1 :, :]
        bias = (weights * float(scale)) * valid.to(weights.dtype)
        tail.scatter_add_(-1, toks, bias.to(tail.dtype))
        return out

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        payload = {
            "format": "flm-core-memory-v1",
            "vocab_size": self.vocab_size,
            "order": self.order,
            "fingerprints": self.fingerprints.cpu(),
            "token_ids": self.token_ids.cpu(),
            "weights": self.weights.cpu(),
            "meta": meta or {},
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, map_location="cpu") -> "CoreMemoryBank":
        obj = torch.load(Path(path), map_location=map_location, weights_only=False)
        if obj.get("format") != "flm-core-memory-v1":
            raise RuntimeError(f"unsupported CoreMemory format: {obj.get('format')!r}")
        return cls(
            obj["fingerprints"],
            obj["token_ids"],
            obj["weights"],
            vocab_size=int(obj["vocab_size"]),
            order=int(obj["order"]),
        )


class CoreMemoryBuilder:
    """Fixed-memory streaming N-gram compiler.

    Uses two deterministic candidate slots per context and a Misra-Gries style
    top-k continuation counter. It never trains model weights.
    """

    def __init__(self, *, vocab_size: int, order: int = 4, slots: int = 1 << 20, top_k: int = 4):
        if not (1 <= order <= 16):
            raise ValueError("order must be in [1,16]")
        if slots < 1024:
            raise ValueError("slots must be >=1024")
        if top_k < 1:
            raise ValueError("top_k must be >=1")
        self.vocab_size = int(vocab_size)
        self.order = int(order)
        self.slots = int(slots)
        self.top_k = int(top_k)
        self.fingerprints = np.full(slots, -1, dtype=np.int64)
        self.tokens = np.full((slots, top_k), -1, dtype=np.int32)
        self.counts = np.zeros((slots, top_k), dtype=np.int32)
        self.accepted_contexts = 0
        self.skipped_collisions = 0
        self.tokens_scanned = 0

    def _slot(self, fp: int) -> int | None:
        a = fp % self.slots
        b = (fp ^ (fp // self.slots)) % self.slots
        if self.fingerprints[a] in (-1, fp):
            return int(a)
        if self.fingerprints[b] in (-1, fp):
            return int(b)
        return None

    def _update(self, fp: int, nxt: int) -> None:
        slot = self._slot(fp)
        if slot is None:
            self.skipped_collisions += 1
            return
        if self.fingerprints[slot] == -1:
            self.fingerprints[slot] = fp
            self.accepted_contexts += 1

        row_t = self.tokens[slot]
        row_c = self.counts[slot]
        matches = np.flatnonzero(row_t == nxt)
        if matches.size:
            row_c[int(matches[0])] += 1
            return
        empty = np.flatnonzero(row_c == 0)
        if empty.size:
            j = int(empty[0])
            row_t[j] = nxt
            row_c[j] = 1
            return
        row_c -= 1
        row_t[row_c == 0] = -1

    def ingest(self, token_ids: Iterable[int]) -> None:
        history: list[int] = []
        for raw in token_ids:
            tok = int(raw)
            if not (0 <= tok < self.vocab_size):
                history.clear()
                continue
            self.tokens_scanned += 1
            if len(history) >= self.order:
                fp = fingerprint_tokens(history[-self.order :])
                self._update(fp, tok)
            history.append(tok)

    def build(self) -> tuple[CoreMemoryBank, CoreMemoryMeta]:
        sums = self.counts.sum(axis=1, keepdims=True).astype(np.float32)
        weights = np.divide(
            self.counts.astype(np.float32),
            np.maximum(sums, 1.0),
            out=np.zeros_like(self.counts, dtype=np.float32),
            where=sums > 0,
        )
        bank = CoreMemoryBank(
            torch.from_numpy(self.fingerprints.copy()),
            torch.from_numpy(self.tokens.copy()),
            torch.from_numpy(weights),
            vocab_size=self.vocab_size,
            order=self.order,
        )
        meta = CoreMemoryMeta(
            vocab_size=self.vocab_size,
            order=self.order,
            slots=self.slots,
            top_k=self.top_k,
            accepted_contexts=self.accepted_contexts,
            skipped_collisions=self.skipped_collisions,
            tokens_scanned=self.tokens_scanned,
        )
        return bank, meta
