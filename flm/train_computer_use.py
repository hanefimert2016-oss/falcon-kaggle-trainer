from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time
import zipfile

from PIL import Image
import numpy as np
import torch

from flm.models import ComputerUseModel, ComputerUseConfig
from flm.runtime import select_runtime

OPS = {"CLICK": 0, "TYPE": 1, "SELECT": 2}

def bytes_fixed(text: str, length: int) -> torch.Tensor:
    payload = list(text.encode("utf-8", "ignore")[:length])
    payload += [0] * (length - len(payload))
    return torch.tensor(payload, dtype=torch.long)

def load_rows(root: Path, image_dir: Path):
    rows = [
        json.loads(line)
        for line in (root / "computer_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not image_dir.exists():
        image_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / "computer_images.zip") as zf:
            zf.extractall(image_dir)
    return rows

def image_tensor(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

def train_computer(data_root: Path, output_root: Path) -> dict:
    runtime = select_runtime()
    smoke = os.environ.get("FLM_SMOKE", "1") == "1"

    cfg = ComputerUseConfig(
        task_len=96 if smoke else 192,
        action_len=64 if smoke else 128,
        image_size=160 if smoke else 224,
        embd=128 if smoke else 384,
        text_layers=1 if smoke else 4,
        vision_layers=1 if smoke else 6,
        n_head=4 if smoke else 8,
    )
    steps = int(os.environ.get("FLM_COMPUTER_STEPS", "25" if smoke else "1500"))
    batch_size = int(os.environ.get("FLM_COMPUTER_BATCH", "4" if smoke else "16"))

    image_dir = output_root / "_computer_images"
    rows = load_rows(data_root, image_dir)
    if not rows:
        raise RuntimeError("computer-use dataset is empty")

    base = ComputerUseModel(cfg).to(runtime.device)
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(base)
    optimizer = torch.optim.AdamW(base.parameters(), lr=3e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")

    out = output_root / "computer_use"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(1337)
    torch.manual_seed(1337)
    start = time.time()
    last = None

    for step in range(steps):
        picks = [random.choice(rows) for _ in range(batch_size)]
        images = torch.stack([
            image_tensor(image_dir / row["image"], cfg.image_size)
            for row in picks
        ]).to(runtime.device)
        task = torch.stack([
            bytes_fixed(row["task"], cfg.task_len) for row in picks
        ]).to(runtime.device)
        actions = torch.stack([
            bytes_fixed(row["action"], cfg.action_len + 1) for row in picks
        ]).to(runtime.device)
        action_in = actions[:, :-1]
        action_target = actions[:, 1:]
        op_target = torch.tensor(
            [OPS.get(str(row["operation"]).upper(), 0) for row in picks],
            device=runtime.device,
        )

        optimizer.zero_grad(set_to_none=True)
        with runtime.autocast():
            _, _, loss = model(
                images, task, action_in, op_target, action_target
            )
            if loss.ndim:
                loss = loss.mean()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        runtime.optimizer_step(optimizer, scaler)
        last = float(loss.detach())

        if step % 5 == 0 or step == steps - 1:
            print(f"computer_use step={step} loss={last:.4f}", flush=True)

    result = {
        "kind": "computer_use",
        "accelerator": runtime.kind,
        "device": str(runtime.device),
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "loss": last,
        "elapsed_s": time.time() - start,
        "examples": len(rows),
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
