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
        for line in (data_root / "coder_train.jsonl").read_text(encoding="utf-8").splitlines():
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
    return torch.tensor(list(raw), dtype=torch.long)

def make_batch(data, batch_size, seq_len, device):
    high = len(data) - seq_len - 1
    if high <= 0:
        raise RuntimeError("dataset too small for configured sequence length")
    starts = torch.randint(0, high, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in starts.tolist()]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts.tolist()]).to(device)
    return x, y

def train_text(kind: str, data_root: Path, output_root: Path) -> dict:
    runtime = select_runtime()
    smoke = os.environ.get("FLM_SMOKE", "1") == "1"

    cfg = TextConfig(
        seq_len=int(os.environ.get("FLM_SEQ_LEN", "192" if smoke else "512")),
        n_layer=int(os.environ.get("FLM_LAYERS", "4" if smoke else "12")),
        n_head=int(os.environ.get("FLM_HEADS", "4" if smoke else "12")),
        n_embd=int(os.environ.get("FLM_EMBD", "256" if smoke else "768")),
    )
    steps = int(os.environ.get(f"FLM_{kind.upper()}_STEPS", "30" if smoke else "2000"))
    batch_size = int(os.environ.get("FLM_BATCH", "16" if smoke else "32"))
    grad_accum = int(os.environ.get("FLM_GRAD_ACCUM", "2"))
    lr = float(os.environ.get("FLM_LR", "3e-4"))

    data = load_text(kind, data_root)
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
            x, y = make_batch(data, batch_size, cfg.seq_len, runtime.device)
            with runtime.autocast():
                _, loss = model(x, y)
                loss = loss.mean() / grad_accum
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

    result = {
        "kind": kind,
        "accelerator": runtime.kind,
        "device": str(runtime.device),
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "loss": last_loss,
        "elapsed_s": time.time() - start,
        "dataset_bytes": int(len(data)),
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
