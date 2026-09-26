#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import torch

from flm.models.text_lm import ByteCausalLM, TextConfig
from flm.models.computer_use_v2 import ComputerUseV2, ComputerUseV2Config
from flm.runtime import select_runtime


OPS = {
    "CLICK": 0,
    "TYPE": 1,
    "KEY": 2,
    "SCROLL": 3,
    "MOVE": 4,
    "DRAG": 5,
    "GAME_ACTION": 6,
    "OTHER": 7,
}


def resolve_v05_root() -> Path:
    configured = os.environ.get("FLM_V05_DATA_ROOT", "").strip()
    candidates: list[Path] = []
    if configured:
        p = Path(configured)
        if (p / "sources.json").is_file():
            candidates.append(p / "sources.json")
        if p.exists():
            candidates.extend(p.rglob("sources.json"))
    root = Path("/kaggle/input")
    if root.exists():
        candidates.extend(root.rglob("sources.json"))
    for manifest in candidates:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("pipeline_version") == "v0.5":
            print(f"resolved_v05_data_root={manifest.parent}", flush=True)
            return manifest.parent
    raise SystemExit("could not locate FLM v0.5 sources.json")


def byte_corpus(path: Path) -> torch.Tensor:
    size = path.stat().st_size
    if size < 4096:
        raise RuntimeError(f"corpus too small: {path} ({size} bytes)")
    return torch.from_file(str(path), shared=False, size=size, dtype=torch.uint8)


def text_batch(data, batch, seq, device):
    high = len(data) - seq - 1
    starts = torch.randint(0, high, (batch,))
    x = torch.stack([data[i:i+seq] for i in starts.tolist()]).long().to(device)
    y = torch.stack([data[i+1:i+seq+1] for i in starts.tolist()]).long().to(device)
    return x, y


@torch.no_grad()
def eval_text(model, data, cfg, runtime, iters=2):
    model.eval()
    vals = []
    for _ in range(max(1, iters)):
        x, y = text_batch(data, 1, cfg.seq_len, runtime.device)
        with runtime.autocast():
            _, loss = model(x, y)
            if loss.ndim:
                loss = loss.mean()
        vals.append(float(loss.detach()))
    model.train()
    return sum(vals) / len(vals)


def train_text_v05(kind, root, out_root, runtime):
    if kind == "main":
        cfg = TextConfig(
            seq_len=int(os.environ.get("FLM_V05_MAIN_SEQ", "512")),
            n_layer=int(os.environ.get("FLM_V05_MAIN_LAYERS", "14")),
            n_head=int(os.environ.get("FLM_V05_MAIN_HEADS", "12")),
            n_embd=int(os.environ.get("FLM_V05_MAIN_EMBD", "768")),
        )
        path = root / "main_train.bin"
        steps = int(os.environ.get("FLM_V05_MAIN_STEPS", "500"))
    else:
        cfg = TextConfig(
            seq_len=int(os.environ.get("FLM_V05_CODER_SEQ", "512")),
            n_layer=int(os.environ.get("FLM_V05_CODER_LAYERS", "12")),
            n_head=int(os.environ.get("FLM_V05_CODER_HEADS", "12")),
            n_embd=int(os.environ.get("FLM_V05_CODER_EMBD", "768")),
        )
        path = root / "coder_train.bin"
        steps = int(os.environ.get("FLM_V05_CODER_STEPS", "400"))

    data = byte_corpus(path)
    split = max(cfg.seq_len * 8, int(len(data) * 0.98))
    split = min(split, len(data) - cfg.seq_len - 2)
    train_data, eval_data = data[:split], data[split:]
    batch = int(os.environ.get("FLM_V05_TEXT_BATCH", "2"))
    accum = int(os.environ.get("FLM_V05_TEXT_ACCUM", "4"))

    base = ByteCausalLM(cfg).to(runtime.device)
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        model = torch.nn.DataParallel(base)
    optimizer = torch.optim.AdamW(base.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    start = time.time()
    last = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            x, y = text_batch(train_data, batch, cfg.seq_len, runtime.device)
            with runtime.autocast():
                _, loss = model(x, y)
                if loss.ndim:
                    loss = loss.mean()
                loss = loss / accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total += float(loss.detach())
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        runtime.optimizer_step(optimizer, scaler)
        last = total
        if step % 10 == 0 or step == steps - 1:
            print(f"v05 {kind} step={step} loss={last:.4f}", flush=True)

    eval_loss = eval_text(
        model, eval_data, cfg, runtime,
        int(os.environ.get("FLM_V05_EVAL_ITERS", "2")),
    )
    result = {
        "kind": kind,
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "train_loss": last,
        "eval_loss": eval_loss,
        "dataset_bytes": int(len(data)),
        "elapsed_s": time.time() - start,
        "config": cfg.__dict__,
    }
    out = out_root / kind
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": base.state_dict(), "config": cfg.__dict__, "result": result}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    del model, base, optimizer
    if runtime.kind == "gpu":
        torch.cuda.empty_cache()
    return result


class ZipImageStore:
    def __init__(self, root: Path):
        self.root = root
        self.handles: dict[str, zipfile.ZipFile] = {}

    def image(self, archive: str, member: str, size: int) -> torch.Tensor:
        if archive not in self.handles:
            self.handles[archive] = zipfile.ZipFile(self.root / archive)
        raw = self.handles[archive].read(member)
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB").resize((size, size))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def bytes_fixed(text, length, pad=258):
    raw = list(str(text).encode("utf-8", "ignore")[:length])
    raw += [pad] * (length - len(raw))
    return torch.tensor(raw, dtype=torch.long)


def action_pair(text, cfg):
    raw = list(str(text).encode("utf-8", "ignore")[: cfg.action_len - 1])
    inp = [cfg.bos_token] + raw
    target = raw + [cfg.eos_token]
    inp += [cfg.pad_token] * (cfg.action_len - len(inp))
    target += [-100] * (cfg.action_len - len(target))
    return torch.tensor(inp), torch.tensor(target)


def cu_batch(rows, store, cfg, domain_to_id, batch, device, rng):
    picks = [rng.choice(rows) for _ in range(batch)]
    images = torch.stack([store.image(r["archive"], r["image"], cfg.image_size) for r in picks]).to(device)
    task = torch.stack([bytes_fixed(r.get("task", ""), cfg.task_len, cfg.pad_token) for r in picks]).to(device)
    pairs = [action_pair(r.get("action", ""), cfg) for r in picks]
    action_in = torch.stack([p[0] for p in pairs]).to(device)
    action_target = torch.stack([p[1] for p in pairs]).to(device)

    op_target = torch.tensor([OPS.get(r.get("operation", "OTHER"), OPS["OTHER"]) for r in picks], device=device)
    op_valid = torch.tensor([bool(r.get("op_valid", False)) for r in picks], dtype=torch.bool, device=device)
    action_valid = torch.tensor([bool(r.get("action_valid", False)) for r in picks], dtype=torch.bool, device=device)

    coord = torch.tensor([r.get("coord") or [0.0, 0.0] for r in picks], dtype=torch.float32, device=device)
    coord_valid = torch.tensor([bool(r.get("coord_valid", False)) for r in picks], dtype=torch.bool, device=device)
    bbox = torch.tensor([r.get("bbox") or [0.0, 0.0, 0.0, 0.0] for r in picks], dtype=torch.float32, device=device)
    bbox_valid = torch.tensor([bool(r.get("bbox_valid", False)) for r in picks], dtype=torch.bool, device=device)

    domain_target = torch.tensor([domain_to_id.get(r.get("domain", ""), 0) for r in picks], device=device)
    domain_valid = torch.tensor([bool(r.get("domain_valid", True)) for r in picks], dtype=torch.bool, device=device)

    kwargs = dict(
        op_target=op_target,
        op_valid=op_valid,
        coord_target=coord,
        coord_valid=coord_valid,
        bbox_target=bbox,
        bbox_valid=bbox_valid,
        domain_target=domain_target,
        domain_valid=domain_valid,
        action_target=action_target,
        action_valid=action_valid,
    )
    return images, task, action_in, kwargs


@torch.no_grad()
def eval_cu(model, rows, store, cfg, domains, runtime):
    rng = random.Random(2027)
    batches = int(os.environ.get("FLM_V05_CU_EVAL_BATCHES", "2"))
    vals = []
    op_ok = op_n = 0
    for _ in range(max(1, batches)):
        images, task, action_in, kw = cu_batch(rows, store, cfg, domains, 1, runtime.device, rng)
        with runtime.autocast():
            out, loss, _ = model(images, task, action_in, **kw)
            if loss is None:
                continue
            if loss.ndim:
                loss = loss.mean()
        vals.append(float(loss.detach()))
        if kw["op_valid"].item():
            op_n += 1
            op_ok += int(out["op_logits"].argmax(-1).item() == kw["op_target"].item())
    model.train()
    return {
        "eval_loss": sum(vals) / max(1, len(vals)),
        "op_accuracy": op_ok / max(1, op_n),
    }


def train_cu_v05(root, out_root, runtime):
    rows = [json.loads(x) for x in (root / "computer_manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 100:
        raise RuntimeError(f"v0.5 computer-use dataset too small: {len(rows)}")
    domains_list = sorted({str(r.get("domain", "")) for r in rows})
    if len(domains_list) >= 128:
        raise RuntimeError(f"too many domains for configured head: {len(domains_list)}")
    domains = {name: i + 1 for i, name in enumerate(domains_list)}

    cfg = ComputerUseV2Config(
        task_len=int(os.environ.get("FLM_V05_CU_TASK_LEN", "192")),
        action_len=int(os.environ.get("FLM_V05_CU_ACTION_LEN", "128")),
        image_size=int(os.environ.get("FLM_V05_CU_IMAGE", "224")),
        embd=int(os.environ.get("FLM_V05_CU_EMBD", "768")),
        text_layers=int(os.environ.get("FLM_V05_CU_TEXT_LAYERS", "4")),
        vision_layers=int(os.environ.get("FLM_V05_CU_VISION_LAYERS", "8")),
        n_head=int(os.environ.get("FLM_V05_CU_HEADS", "12")),
    )
    steps = int(os.environ.get("FLM_V05_CU_STEPS", "300"))
    batch = int(os.environ.get("FLM_V05_CU_BATCH", "2"))
    rng = random.Random(1337)
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * 0.98))
    train_rows, eval_rows = rows[:cut], rows[cut:] or rows[-8:]
    store = ZipImageStore(root)

    base = ComputerUseV2(cfg).to(runtime.device)
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        model = torch.nn.DataParallel(base)
    optimizer = torch.optim.AdamW(base.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    start = time.time()
    last = 0.0
    for step in range(steps):
        images, task, action_in, kw = cu_batch(train_rows, store, cfg, domains, batch, runtime.device, rng)
        optimizer.zero_grad(set_to_none=True)
        with runtime.autocast():
            _, loss, parts = model(images, task, action_in, **kw)
            if loss is None:
                raise RuntimeError("no supervised loss in sampled computer-use batch")
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
        if step % 10 == 0 or step == steps - 1:
            part_vals = {k: float(v.detach().mean()) for k, v in parts.items()}
            print(f"v05 computer_use step={step} loss={last:.4f} parts={part_vals}", flush=True)

    ev = eval_cu(model, eval_rows, store, cfg, domains, runtime)
    result = {
        "kind": "computer_use",
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "train_loss": last,
        **ev,
        "examples": len(rows),
        "domains": len(domains_list),
        "domain_names": domains_list,
        "elapsed_s": time.time() - start,
        "config": cfg.__dict__,
    }
    out = out_root / "computer_use"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": base.state_dict(), "config": cfg.__dict__, "domains": domains, "result": result}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def main() -> int:
    root = resolve_v05_root()
    out = Path(os.environ.get("FLM_V05_OUTPUT_ROOT", "/kaggle/working/flm-v0.5"))
    out.mkdir(parents=True, exist_ok=True)
    runtime = select_runtime("gpu" if os.environ.get("FLM_ACCELERATOR", "gpu") == "gpu" else None)
    random.seed(1337)
    torch.manual_seed(1337)
    if runtime.kind == "gpu":
        torch.cuda.manual_seed_all(1337)

    source_manifest = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    (out / "sources.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    results = [
        train_text_v05("main", root, out, runtime),
        train_cu_v05(root, out, runtime),
        train_text_v05("coder", root, out, runtime),
    ]
    summary = {"models": results, "pipeline_version": "v0.5", "accelerator": runtime.kind}
    (out / "suite_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"v05_suite_complete output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
