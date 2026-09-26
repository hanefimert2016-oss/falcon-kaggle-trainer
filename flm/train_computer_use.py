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

OP_NAMES = ("CLICK", "TYPE", "KEY", "SCROLL", "MOVE", "OTHER")
OPS = {name: i for i, name in enumerate(OP_NAMES)}


def task_bytes(text: str, cfg: ComputerUseConfig) -> torch.Tensor:
    payload = list(text.encode("utf-8", "ignore")[: cfg.task_len])
    payload += [cfg.pad_token] * (cfg.task_len - len(payload))
    return torch.tensor(payload, dtype=torch.long)


def action_bytes(
    text: str, cfg: ComputerUseConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    # Teacher forcing: [BOS, a, b] -> [a, b, EOS], masked after EOS.
    raw = list(text.encode("utf-8", "ignore")[: cfg.action_len - 1])
    action_in = [cfg.bos_token] + raw
    target = raw + [cfg.eos_token]
    action_in += [cfg.pad_token] * (cfg.action_len - len(action_in))
    target += [-100] * (cfg.action_len - len(target))
    return (
        torch.tensor(action_in, dtype=torch.long),
        torch.tensor(target, dtype=torch.long),
    )


def load_rows(root: Path, image_dir: Path):
    rows = [
        json.loads(line)
        for line in (root / "computer_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
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
        # Normalize around 0 for a from-scratch vision encoder.
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def make_batch(rows, image_dir, cfg, batch_size, device, rng):
    picks = [rng.choice(rows) for _ in range(batch_size)]
    images = torch.stack(
        [image_tensor(image_dir / row["image"], cfg.image_size) for row in picks]
    ).to(device)
    tasks = torch.stack([task_bytes(row["task"], cfg) for row in picks]).to(device)
    pairs = [action_bytes(row["action"], cfg) for row in picks]
    action_in = torch.stack([x[0] for x in pairs]).to(device)
    action_target = torch.stack([x[1] for x in pairs]).to(device)
    op_target = torch.tensor(
        [OPS.get(str(row["operation"]).upper(), OPS["OTHER"]) for row in picks],
        device=device,
    )
    return images, tasks, action_in, op_target, action_target


@torch.no_grad()
def evaluate(model, rows, image_dir, cfg, device, runtime) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    rng = random.Random(2026)
    batches = min(4, max(1, len(rows)))
    for _ in range(batches):
        images, task, action_in, op_target, action_target = make_batch(
            rows, image_dir, cfg, 1, device, rng
        )
        with runtime.autocast():
            op_logits, _, loss = model(
                images, task, action_in, op_target, action_target
            )
            if loss.ndim:
                loss = loss.mean()
        total_loss += float(loss.detach())
        total_correct += int((op_logits.argmax(dim=-1) == op_target).sum().item())
        total += int(op_target.numel())
    model.train()
    return {
        "eval_loss": total_loss / batches,
        "op_accuracy": total_correct / max(1, total),
    }


def train_computer(data_root: Path, output_root: Path) -> dict:
    runtime = select_runtime()
    smoke = os.environ.get("FLM_SMOKE", "1") == "1"

    cfg = ComputerUseConfig(
        task_len=int(os.environ.get("FLM_COMPUTER_TASK_LEN", "128" if smoke else "256")),
        action_len=int(os.environ.get("FLM_COMPUTER_ACTION_LEN", "96" if smoke else "192")),
        image_size=int(os.environ.get("FLM_COMPUTER_IMAGE_SIZE", "160" if smoke else "224")),
        embd=int(os.environ.get("FLM_COMPUTER_EMBD", "192" if smoke else "384")),
        text_layers=int(os.environ.get("FLM_COMPUTER_TEXT_LAYERS", "2" if smoke else "4")),
        vision_layers=int(os.environ.get("FLM_COMPUTER_VISION_LAYERS", "2" if smoke else "6")),
        n_head=int(os.environ.get("FLM_COMPUTER_HEADS", "6" if smoke else "8")),
    )
    steps = int(os.environ.get("FLM_COMPUTER_STEPS", "20" if smoke else "1500"))
    batch_size = int(os.environ.get("FLM_COMPUTER_BATCH", "4" if smoke else "16"))

    image_dir = output_root / "_computer_images"
    rows = load_rows(data_root, image_dir)
    if len(rows) < 2:
        raise RuntimeError("computer-use dataset needs at least two real samples")

    # Deterministic holdout for accelerator-side evaluation.
    rng = random.Random(1337)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.8))
    train_rows = shuffled[:cut]
    eval_rows = shuffled[cut:] or shuffled[-1:]

    base = ComputerUseModel(cfg).to(runtime.device)
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(base)

    optimizer = torch.optim.AdamW(base.parameters(), lr=3e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")

    out = output_root / "computer_use"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)
    start = time.time()
    last = None

    for step in range(steps):
        images, task, action_in, op_target, action_target = make_batch(
            train_rows,
            image_dir,
            cfg,
            batch_size,
            runtime.device,
            rng,
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

    eval_result = evaluate(
        model, eval_rows, image_dir, cfg, runtime.device, runtime
    )
    result = {
        "kind": "computer_use",
        "accelerator": runtime.kind,
        "device": str(runtime.device),
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "train_loss": last,
        **eval_result,
        "elapsed_s": time.time() - start,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "vision": {
            "image_size": cfg.image_size,
            "patch": cfg.patch,
            "vision_layers": cfg.vision_layers,
        },
        "operation_classes": list(OP_NAMES),
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
