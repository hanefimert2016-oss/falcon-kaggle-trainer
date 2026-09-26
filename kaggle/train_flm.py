"""FLM v0.2: from-scratch byte-level causal language model trainer.

Runs on Kaggle GPU compute and intentionally uses no pretrained model or tokenizer.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 256
    seq_len: int = int(os.environ.get("FLM_SEQ_LEN", "256"))
    n_layer: int = int(os.environ.get("FLM_LAYERS", "8"))
    n_head: int = int(os.environ.get("FLM_HEADS", "8"))
    n_embd: int = int(os.environ.get("FLM_EMBD", "512"))
    dropout: float = 0.0
    batch_size: int = int(os.environ.get("FLM_BATCH", "24"))
    grad_accum: int = int(os.environ.get("FLM_GRAD_ACCUM", "4"))
    max_steps: int = int(os.environ.get("FLM_STEPS", "1200"))
    lr: float = float(os.environ.get("FLM_LR", "3e-4"))
    weight_decay: float = 0.1
    warmup_steps: int = 50
    eval_interval: int = 100
    eval_iters: int = 20
    seed: int = 1337


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class FLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        if t > self.cfg.seq_len:
            raise ValueError(f"sequence length {t} > configured {self.cfg.seq_len}")
        pos = torch.arange(0, t, device=idx.device)
        x = self.token(idx) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def load_bytes() -> torch.Tensor:
    input_root = Path("/kaggle/input")
    candidates = sorted(input_root.rglob("*.txt")) if input_root.exists() else []
    chunks: list[bytes] = []
    for p in candidates[:64]:
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if data:
            chunks.append(data[:64 * 1024 * 1024])
    if not chunks:
        fallback = (
            "Falcon Language Model is trained from scratch. "
            "Bu veri yalnizca egitim hattini test etmek icindir.\n"
        ) * 10000
        chunks = [fallback.encode("utf-8")]
    raw = b"\n".join(chunks)
    return torch.tensor(list(raw), dtype=torch.long)


def get_batch(data: torch.Tensor, cfg: Config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - cfg.seq_len - 1
    if max_start <= 0:
        raise RuntimeError("dataset is smaller than seq_len + 1")
    starts = torch.randint(0, max_start, (cfg.batch_size,))
    x = torch.stack([data[i:i + cfg.seq_len] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + cfg.seq_len + 1] for i in starts.tolist()])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_for(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, ratio)))


@torch.no_grad()
def evaluate(model: nn.Module, data: torch.Tensor, cfg: Config, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg, device)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("model returned no loss during evaluation")
        if loss.ndim:
            loss = loss.mean()
        losses.append(float(loss.detach()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    cfg = Config()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")

    out = Path("/kaggle/working/flm-v0.2")
    out.mkdir(parents=True, exist_ok=True)
    data = load_bytes()
    split = max(cfg.seq_len + 2, int(len(data) * 0.98))
    split = min(split, len(data) - cfg.seq_len - 2)
    train_data = data[:split]
    val_data = data[split:]
    if len(val_data) <= cfg.seq_len + 1:
        val_data = train_data[-max(cfg.seq_len * 8, cfg.seq_len + 2):]

    base_model = FLM(cfg).to(device)
    params = sum(p.numel() for p in base_model.parameters())
    print(json.dumps({
        "device": str(device),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "parameters": params,
        "dataset_bytes": int(len(data)),
        "config": asdict(cfg),
    }, ensure_ascii=False), flush=True)

    model: nn.Module = base_model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model)

    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = time.time()

    model.train()
    for step in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_data, cfg, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("model returned no loss during training")
                if loss.ndim:
                    loss = loss.mean()
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            running += float(loss.detach())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
        lr = lr_for(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            elapsed = time.time() - start
            print(f"step={step} train_loss={running:.4f} lr={lr:.6g} elapsed_s={elapsed:.1f}", flush=True)
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            val = evaluate(model, val_data, cfg, device)
            print(f"step={step} val_loss={val:.4f}", flush=True)
            torch.save({
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": asdict(cfg),
            }, out / "checkpoint.pt")
            (out / "metrics.json").write_text(json.dumps({
                "step": step,
                "train_loss": running,
                "val_loss": val,
                "parameters": params,
                "elapsed_s": time.time() - start,
            }, indent=2), encoding="utf-8")

    print(f"training_complete output={out}", flush=True)


if __name__ == "__main__":
    main()
