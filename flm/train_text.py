from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import torch

from flm.models import ByteCausalLM, TextConfig
from flm.runtime import select_runtime


def load_text(kind: str, data_root: Path) -> torch.Tensor:
    if kind == "main":
        raw = (data_root / "main_train.txt").read_bytes()
    elif kind == "coder":
        chunks = []
        for line in (
            data_root / "coder_train.jsonl"
        ).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(
                f"<instruction>\n{row['instruction']}\n</instruction>\n"
                f"<code>\n{row['code']}\n</code>\n"
            )
        raw = "\n".join(chunks).encode("utf-8")
    else:
        raise ValueError(kind)
    if len(raw) < 1024:
        raise RuntimeError(f"{kind} dataset unexpectedly small: {len(raw)} bytes")
    return torch.tensor(list(raw), dtype=torch.long)


def make_batch(data, batch_size, seq_len, device):
    high = len(data) - seq_len - 1
    if high <= 0:
        raise RuntimeError("dataset too small for configured sequence length")
    starts = torch.randint(0, high, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in starts.tolist()]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts.tolist()]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, cfg, runtime, iters=4):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = make_batch(data, 1, cfg.seq_len, runtime.device)
        with runtime.autocast():
            _, loss = model(x, y)
            if loss.ndim:
                loss = loss.mean()
        losses.append(float(loss.detach()))
    model.train()
    return sum(losses) / len(losses)


def train_text(kind: str, data_root: Path, output_root: Path) -> dict:
    runtime = select_runtime()
    smoke = os.environ.get("FLM_SMOKE", "1") == "1"

    # Main and coder are separate weights. Their first architecture is the same
    # decoder family, but each is independently pretrained on its own real data.
    default_seq = "256" if smoke else ("1024" if kind == "coder" else "768")
    default_layers = "8" if smoke else "12"
    default_heads = "8" if smoke else "12"
    default_embd = "512" if smoke else "768"

    cfg = TextConfig(
        seq_len=int(os.environ.get(f"FLM_{kind.upper()}_SEQ_LEN", default_seq)),
        n_layer=int(os.environ.get(f"FLM_{kind.upper()}_LAYERS", default_layers)),
        n_head=int(os.environ.get(f"FLM_{kind.upper()}_HEADS", default_heads)),
        n_embd=int(os.environ.get(f"FLM_{kind.upper()}_EMBD", default_embd)),
    )
    steps = int(
        os.environ.get(
            f"FLM_{kind.upper()}_STEPS", "25" if smoke else "2000"
        )
    )
    batch_size = int(os.environ.get("FLM_BATCH", "8" if smoke else "32"))
    grad_accum = int(os.environ.get("FLM_GRAD_ACCUM", "2"))
    lr = float(os.environ.get("FLM_LR", "3e-4"))

    data = load_text(kind, data_root)
    split = min(len(data) - cfg.seq_len - 2, max(cfg.seq_len + 2, int(len(data) * 0.95)))
    train_data = data[:split]
    eval_data = data[split:]
    if len(eval_data) <= cfg.seq_len + 1:
        eval_data = train_data[-max(cfg.seq_len * 8, cfg.seq_len + 2) :]

    model = ByteCausalLM(cfg).to(runtime.device)
    base = model
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        base.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    out = output_root / kind
    out.mkdir(parents=True, exist_ok=True)

    random.seed(1337)
    torch.manual_seed(1337)
    start = time.time()
    last_loss = None

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(grad_accum):
            x, y = make_batch(train_data, batch_size, cfg.seq_len, runtime.device)
            with runtime.autocast():
                _, loss = model(x, y)
                if loss.ndim:
                    loss = loss.mean()
                loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.detach())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        runtime.optimizer_step(optimizer, scaler)
        last_loss = running

        if step % 5 == 0 or step == steps - 1:
            print(f"{kind} step={step} loss={running:.4f}", flush=True)

    # Test/eval stays on the selected GPU or TPU.
    eval_loss = evaluate(model, eval_data, cfg, runtime)
    result = {
        "kind": kind,
        "accelerator": runtime.kind,
        "device": str(runtime.device),
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "train_loss": last_loss,
        "eval_loss": eval_loss,
        "elapsed_s": time.time() - start,
        "dataset_bytes": int(len(data)),
        "train_bytes": int(len(train_data)),
        "eval_bytes": int(len(eval_data)),
        "config": cfg.__dict__,
    }
    torch.save(
        {"model": base.state_dict(), "config": cfg.__dict__, "result": result},
        out / "checkpoint.pt",
    )
    (out / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)
    return result
