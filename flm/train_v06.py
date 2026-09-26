#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from flm.models.text_lm import ByteCausalLM, TextConfig
from flm.models.computer_use_v2 import ComputerUseV2, ComputerUseV2Config
from flm.runtime import select_runtime
from flm.train_v05 import ZipImageStore, bytes_fixed, action_pair, OPS, eval_cu


def resolve_pipeline_root(version: str) -> Path:
    configured = os.environ.get("FLM_V06_DATA_ROOT" if version == "v0.6" else "FLM_V05_DATA_ROOT", "").strip()
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
    seen = set()
    for manifest in candidates:
        key = str(manifest.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("pipeline_version") == version:
            print(f"resolved_{version}_data_root={manifest.parent}", flush=True)
            return manifest.parent
    raise SystemExit(f"could not locate FLM {version} sources.json")


def mmap_tokens(path: Path) -> np.memmap:
    if not path.is_file() or path.stat().st_size < 4096:
        raise RuntimeError(f"missing/small token file: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def mmap_mask(path: Path) -> np.memmap:
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"missing/small mask file: {path}")
    return np.memmap(path, dtype=np.uint8, mode="r")


def make_xy(data: np.memmap, block_ids: np.ndarray, seq: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for block in block_ids.tolist():
        s = int(block) * seq
        xs.append(np.asarray(data[s:s+seq], dtype=np.int64))
        ys.append(np.asarray(data[s+1:s+seq+1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs)).long().to(device, non_blocking=True)
    y = torch.from_numpy(np.stack(ys)).long().to(device, non_blocking=True)
    return x, y


def make_sft_xy(tokens: np.memmap, mask: np.memmap, block_ids: np.ndarray, seq: int, device):
    x, y = make_xy(tokens, block_ids, seq, device)
    ms = []
    for block in block_ids.tolist():
        s = int(block) * seq
        ms.append(np.asarray(mask[s+1:s+seq+1], dtype=np.uint8))
    m = torch.from_numpy(np.stack(ms)).bool().to(device, non_blocking=True)
    y = y.masked_fill(~m, -100)
    return x, y, int(m.sum().item())


def cosine_lr(step: int, total: int, peak: float, warmup: int, floor_ratio: float = 0.1) -> float:
    if total <= 1:
        return peak
    if step < warmup:
        return peak * max(0.05, (step + 1) / max(1, warmup))
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * p)))


@torch.no_grad()
def eval_token_model(model, data, seq, runtime, iters=8):
    model.eval()
    vals = []
    rng = np.random.default_rng(2026)
    n_blocks = max(1, (len(data) - 1) // seq)
    for _ in range(max(1, iters)):
        block = int(rng.integers(0, n_blocks))
        x, y = make_xy(data, np.asarray([block]), seq, runtime.device)
        with runtime.autocast():
            _, loss = model(x, y)
            if loss.ndim:
                loss = loss.mean()
        vals.append(float(loss.detach()))
    model.train()
    return sum(vals) / len(vals)


def build_text_model(kind: str, vocab_size: int, seq: int):
    if kind == "main":
        cfg = TextConfig(vocab_size=vocab_size, seq_len=seq, n_layer=14, n_head=12, n_embd=768)
    elif kind == "coder":
        cfg = TextConfig(vocab_size=vocab_size, seq_len=seq, n_layer=12, n_head=12, n_embd=768)
    else:
        raise ValueError(kind)
    return ByteCausalLM(cfg), cfg


def train_pretrain(kind, data, out_root, runtime, vocab_size, epochs, batch, accum, max_steps=0):
    seq = int(os.environ.get("FLM_V06_SEQ", "512"))
    base, cfg = build_text_model(kind, vocab_size, seq)
    base = base.to(runtime.device)
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        model = torch.nn.DataParallel(base)

    split = int(len(data) * 0.99)
    split -= split % seq
    train_data = data[:split]
    eval_data = data[split:]
    n_blocks = (len(train_data) - 1) // seq
    blocks_per_step = batch * accum
    steps_per_epoch = max(1, n_blocks // blocks_per_step)
    total_steps = steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    peak_lr = 2.5e-4 if kind == "main" else 2.8e-4
    warmup = min(500, max(10, total_steps // 25))
    optimizer = torch.optim.AdamW(base.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    rng = np.random.default_rng(1337 if kind == "main" else 7331)
    global_step = 0
    last = 0.0
    start = time.time()
    tokens_seen = 0

    for epoch in range(epochs):
        order = rng.permutation(n_blocks)
        pos = 0
        while pos + blocks_per_step <= len(order) and global_step < total_steps:
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(accum):
                ids = order[pos:pos+batch]
                pos += batch
                x, y = make_xy(train_data, ids, seq, runtime.device)
                with runtime.autocast():
                    _, loss = model(x, y)
                    if loss.ndim:
                        loss = loss.mean()
                    scaled = loss / accum
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                total_loss += float(scaled.detach())
                tokens_seen += int(x.numel())
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            lr = cosine_lr(global_step, total_steps, peak_lr, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            runtime.optimizer_step(optimizer, scaler)
            last = total_loss
            if global_step % 100 == 0 or global_step == total_steps - 1:
                print(f"v06 {kind} pretrain step={global_step}/{total_steps} epoch={epoch+1}/{epochs} loss={last:.4f} lr={lr:.7f} tokens_seen={tokens_seen}", flush=True)
            global_step += 1
        if global_step >= total_steps:
            break

    ev = eval_token_model(model, eval_data, seq, runtime, int(os.environ.get("FLM_V06_EVAL_ITERS", "8")))
    result = {
        "kind": kind,
        "stage": "pretrain",
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": global_step,
        "epochs_requested": epochs,
        "train_tokens": int(len(train_data)),
        "eval_tokens": int(len(eval_data)),
        "tokens_seen": tokens_seen,
        "train_loss": last,
        "eval_loss": ev,
        "elapsed_s": time.time() - start,
        "config": cfg.__dict__,
    }
    print(json.dumps(result), flush=True)
    return base, cfg, result


def train_sft(base, cfg, tokens, mask, runtime, epochs, batch, accum, max_steps=0):
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        model = torch.nn.DataParallel(base)
    seq = cfg.seq_len
    n_blocks = (len(tokens) - 1) // seq
    eligible = np.asarray(
        [i for i in range(n_blocks) if np.asarray(mask[i*seq+1:i*seq+seq+1]).any()],
        dtype=np.int64,
    )
    if len(eligible) < 100:
        raise RuntimeError(f"too few supervised SFT blocks: {len(eligible)}")
    blocks_per_step = batch * accum
    steps_per_epoch = max(1, len(eligible) // blocks_per_step)
    total_steps = steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    peak_lr = 8e-5
    warmup = min(250, max(10, total_steps // 30))
    optimizer = torch.optim.AdamW(base.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    rng = np.random.default_rng(606)
    global_step = 0
    last = 0.0
    supervised_seen = 0
    start = time.time()

    for epoch in range(epochs):
        order = rng.permutation(eligible)
        pos = 0
        while pos + blocks_per_step <= len(order) and global_step < total_steps:
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            micro = 0
            while micro < accum and pos + batch <= len(order):
                ids = order[pos:pos+batch]
                pos += batch
                x, y, sup = make_sft_xy(tokens, mask, ids, seq, runtime.device)
                if sup <= 0:
                    continue
                with runtime.autocast():
                    _, loss = model(x, y)
                    if loss.ndim:
                        loss = loss.mean()
                    scaled = loss / accum
                if not torch.isfinite(scaled):
                    raise RuntimeError(f"non-finite SFT loss at step {global_step}")
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                total_loss += float(scaled.detach())
                supervised_seen += sup
                micro += 1
            if micro == 0:
                continue
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            lr = cosine_lr(global_step, total_steps, peak_lr, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            runtime.optimizer_step(optimizer, scaler)
            last = total_loss
            if global_step % 100 == 0 or global_step == total_steps - 1:
                print(f"v06 main sft step={global_step}/{total_steps} epoch={epoch+1}/{epochs} loss={last:.4f} lr={lr:.7f} supervised_seen={supervised_seen}", flush=True)
            global_step += 1
        if global_step >= total_steps:
            break

    result = {
        "stage": "sft",
        "steps": global_step,
        "epochs_requested": epochs,
        "eligible_blocks": int(len(eligible)),
        "supervised_tokens_seen": supervised_seen,
        "train_loss": last,
        "elapsed_s": time.time() - start,
    }
    print(json.dumps(result), flush=True)
    return base, result


def cu_batch_rows(picks, store, cfg, domain_to_id, device):
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
    kw = dict(
        op_target=op_target, op_valid=op_valid,
        coord_target=coord, coord_valid=coord_valid,
        bbox_target=bbox, bbox_valid=bbox_valid,
        domain_target=domain_target, domain_valid=domain_valid,
        action_target=action_target, action_valid=action_valid,
    )
    return images, task, action_in, kw


def train_cu_v06(root, out_root, runtime, epochs=2, max_steps=0):
    rows = [json.loads(x) for x in (root/"computer_manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 6000:
        raise RuntimeError(f"expected expanded computer-use data, found {len(rows)}")
    domains_list = sorted({str(r.get("domain", "")) for r in rows})
    domains = {name: i + 1 for i, name in enumerate(domains_list)}
    cfg = ComputerUseV2Config(
        task_len=192, action_len=128, image_size=224,
        embd=768, text_layers=4, vision_layers=8, n_head=12,
    )
    rng = random.Random(606)
    rng.shuffle(rows)
    cut = int(len(rows) * 0.98)
    train_rows, eval_rows = rows[:cut], rows[cut:]
    batch = int(os.environ.get("FLM_V06_CU_BATCH", "2"))
    steps_per_epoch = len(train_rows) // batch
    total_steps = steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)

    store = ZipImageStore(root)
    base = ComputerUseV2(cfg).to(runtime.device)
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        model = torch.nn.DataParallel(base)
    optimizer = torch.optim.AdamW(base.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    start = time.time()
    global_step = 0
    images_seen = 0
    last = 0.0

    for epoch in range(epochs):
        rng.shuffle(train_rows)
        for pos in range(0, len(train_rows) - batch + 1, batch):
            if global_step >= total_steps:
                break
            picks = train_rows[pos:pos+batch]
            images, task, action_in, kw = cu_batch_rows(picks, store, cfg, domains, runtime.device)
            optimizer.zero_grad(set_to_none=True)
            with runtime.autocast():
                _, loss, parts = model(images, task, action_in, **kw)
                if loss.ndim:
                    loss = loss.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite CU loss at step {global_step}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            lr = cosine_lr(global_step, total_steps, 1.5e-4, min(250, max(10, total_steps//30)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            runtime.optimizer_step(optimizer, scaler)
            last = float(loss.detach())
            images_seen += len(picks)
            if global_step % 100 == 0 or global_step == total_steps - 1:
                pv = {k: float(v.detach().mean()) for k, v in parts.items()}
                print(f"v06 computer_use step={global_step}/{total_steps} epoch={epoch+1}/{epochs} images_seen={images_seen} loss={last:.4f} parts={pv}", flush=True)
            global_step += 1
        if global_step >= total_steps:
            break

    ev = eval_cu(model, eval_rows, store, cfg, domains, runtime)
    result = {
        "kind": "computer_use",
        "pipeline_version": "v0.6",
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": global_step,
        "epochs_requested": epochs,
        "images_seen": images_seen,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "examples": len(rows),
        "domains": len(domains_list),
        "domain_names": domains_list,
        "train_loss": last,
        **ev,
        "elapsed_s": time.time() - start,
        "config": cfg.__dict__,
    }
    out = out_root/"computer_use"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": base.state_dict(), "config": cfg.__dict__, "domains": domains, "result": result}, out/"checkpoint.pt")
    (out/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    del model, base, optimizer
    if runtime.kind == "gpu":
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def chat_generate(model, cfg, tok: Tokenizer, prompt: str, device, max_new=120):
    model.eval()
    bos = tok.token_to_id("<bos>")
    user = tok.token_to_id("<|user|>")
    assistant = tok.token_to_id("<|assistant|>")
    end = tok.token_to_id("<|end|>")
    prefix = [bos, user] + tok.encode(prompt, add_special_tokens=False).ids + [assistant]
    ids = list(prefix)
    g = torch.Generator(device=device)
    g.manual_seed(606)
    for _ in range(max_new):
        x = torch.tensor([ids[-cfg.seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1].float()
        topv, topi = torch.topk(next_logits, k=min(40, next_logits.numel()))
        probs = torch.softmax(topv / 0.75, dim=-1)
        pick = int(topi[torch.multinomial(probs, 1, generator=g)])
        if pick == end:
            break
        ids.append(pick)
    generated = ids[len(prefix):]
    return tok.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    text_root = resolve_pipeline_root("v0.6")
    cu_root = resolve_pipeline_root("v0.5")
    out = Path(os.environ.get("FLM_V06_OUTPUT_ROOT", "/kaggle/working/flm-v0.6-full"))
    out.mkdir(parents=True, exist_ok=True)
    runtime = select_runtime("gpu")
    random.seed(606)
    torch.manual_seed(606)
    if runtime.kind == "gpu":
        torch.cuda.manual_seed_all(606)

    tok = Tokenizer.from_file(str(text_root/"tokenizer.json"))
    vocab_size = tok.get_vocab_size()
    if vocab_size < 4096:
        raise RuntimeError(f"v0.6 tokenizer too small: {vocab_size}")

    main_data = mmap_tokens(text_root/"main_train.u16")
    coder_data = mmap_tokens(text_root/"coder_train.u16")
    sft_tokens = mmap_tokens(text_root/"main_sft_tokens.u16")
    sft_mask = mmap_mask(text_root/"main_sft_mask.u8")
    if len(sft_tokens) != len(sft_mask):
        raise RuntimeError("SFT token/mask length mismatch")

    batch = int(os.environ.get("FLM_V06_TEXT_BATCH", "4"))
    accum = int(os.environ.get("FLM_V06_TEXT_ACCUM", "2"))
    main_epochs = int(os.environ.get("FLM_V06_MAIN_EPOCHS", "2"))
    sft_epochs = int(os.environ.get("FLM_V06_SFT_EPOCHS", "1"))
    coder_epochs = int(os.environ.get("FLM_V06_CODER_EPOCHS", "1"))
    cu_epochs = int(os.environ.get("FLM_V06_CU_EPOCHS", "2"))
    max_pre = int(os.environ.get("FLM_V06_MAX_PRETRAIN_STEPS", "0"))
    max_sft = int(os.environ.get("FLM_V06_MAX_SFT_STEPS", "0"))
    max_cu = int(os.environ.get("FLM_V06_MAX_CU_STEPS", "0"))

    main, main_cfg, main_pre = train_pretrain(
        "main", main_data, out, runtime, vocab_size, main_epochs, batch, accum, max_pre
    )
    main, sft_result = train_sft(
        main, main_cfg, sft_tokens, sft_mask, runtime, sft_epochs, batch, accum, max_sft
    )

    prompts = [
        "Merhaba, nasılsın?",
        "Türkiye'nin başkenti neresidir?",
        "Bugün kendimi biraz yorgun hissediyorum. Bana kısa bir öneri ver.",
        "Hello! How are you today?",
    ]
    samples = [{"prompt": p, "output": chat_generate(main, main_cfg, tok, p, runtime.device)} for p in prompts]
    print("V06_QUALITY_SAMPLES=" + json.dumps(samples, ensure_ascii=False), flush=True)

    main_out = out/"main"
    main_out.mkdir(parents=True, exist_ok=True)
    main_result = {
        **main_pre,
        "stage": "pretrain+sft",
        "sft": sft_result,
        "quality_samples": samples,
    }
    torch.save({"model": main.state_dict(), "config": main_cfg.__dict__, "result": main_result}, main_out/"checkpoint.pt")
    (main_out/"metrics.json").write_text(json.dumps(main_result, indent=2, ensure_ascii=False), encoding="utf-8")
    del main
    if runtime.kind == "gpu":
        torch.cuda.empty_cache()

    cu_result = train_cu_v06(cu_root, out, runtime, cu_epochs, max_cu)

    coder, coder_cfg, coder_result = train_pretrain(
        "coder", coder_data, out, runtime, vocab_size, coder_epochs, batch, accum, max_pre
    )
    coder_out = out/"coder"
    coder_out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": coder.state_dict(), "config": coder_cfg.__dict__, "result": coder_result}, coder_out/"checkpoint.pt")
    (coder_out/"metrics.json").write_text(json.dumps(coder_result, indent=2), encoding="utf-8")

    shutil.copy2(text_root/"tokenizer.json", out/"tokenizer.json")
    shutil.copy2(text_root/"tokenizer_meta.json", out/"tokenizer_meta.json")
    shutil.copy2(text_root/"sources.json", out/"text_sources.json")
    shutil.copy2(cu_root/"sources.json", out/"computer_sources.json")

    summary = {
        "pipeline_version": "v0.6",
        "accelerator": runtime.kind,
        "tokenizer_vocab_size": vocab_size,
        "models": [main_result, cu_result, coder_result],
        "quality_samples": samples,
    }
    (out/"suite_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out/"quality_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"v06_suite_complete output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
