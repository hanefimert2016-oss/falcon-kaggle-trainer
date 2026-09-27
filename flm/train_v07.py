#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

from flm.models.text_lm import ByteCausalLM, TextConfig
from flm.models.computer_use_v3 import ComputerUseV3, ComputerUseV3Config
from flm.computer_use.context import build_context_ids, summarize_action
from flm.runtime import select_runtime
from flm.train_v05 import ZipImageStore


OPS_V3 = {
    "MOVE": 0,
    "CLICK": 1,
    "DOUBLE_CLICK": 2,
    "RIGHT_CLICK": 3,
    "DRAG": 4,
    "SCROLL": 5,
    "TYPE": 6,
    "KEY": 7,
    "KEY_DOWN": 8,
    "KEY_UP": 9,
    "WAIT": 10,
    "DONE": 11,
}
ID_TO_OP = {v: k for k, v in OPS_V3.items()}


def resolve_pipeline_root(version: str) -> Path:
    env_name = "FLM_V07_DATA_ROOT" if version == "v0.7" else "FLM_V05_DATA_ROOT"
    configured = os.environ.get(env_name, "").strip()
    candidates: list[Path] = []
    if configured:
        p = Path(configured)
        if (p / "sources.json").is_file():
            candidates.append(p / "sources.json")
        if p.exists():
            candidates.extend(p.rglob("sources.json"))
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        candidates.extend(kaggle.rglob("sources.json"))
    for manifest in candidates:
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
        raise RuntimeError(f"token file missing/small: {path}")
    return np.memmap(path, mode="r", dtype="<u2")


def mmap_mask(path: Path) -> np.memmap:
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"mask file missing/small: {path}")
    return np.memmap(path, mode="r", dtype=np.uint8)


def make_xy(data, starts: np.ndarray, seq: int, device):
    x = np.stack([np.asarray(data[s:s + seq], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(data[s + 1:s + seq + 1], dtype=np.int64) for s in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def sample_starts(rng: np.random.Generator, length: int, seq: int, batch: int) -> np.ndarray:
    high = length - seq - 2
    if high <= 0:
        raise RuntimeError(f"dataset too short for seq={seq}: {length}")
    return rng.integers(0, high, size=batch, dtype=np.int64)


def make_sft_batch(data, mask, rng, seq, batch, device, min_supervised=8):
    high = len(data) - seq - 2
    xs, ys = [], []
    supervised = 0
    attempts = 0
    while len(xs) < batch and attempts < batch * 200:
        attempts += 1
        s = int(rng.integers(0, high))
        target_mask = np.asarray(mask[s + 1:s + seq + 1], dtype=np.bool_)
        if int(target_mask.sum()) < min_supervised:
            continue
        x = np.asarray(data[s:s + seq], dtype=np.int64)
        y = np.asarray(data[s + 1:s + seq + 1], dtype=np.int64).copy()
        y[~target_mask] = -100
        xs.append(x)
        ys.append(y)
        supervised += int(target_mask.sum())
    if len(xs) != batch:
        raise RuntimeError(f"could not sample SFT batch: got {len(xs)}/{batch}")
    return (
        torch.from_numpy(np.stack(xs)).to(device),
        torch.from_numpy(np.stack(ys)).to(device),
        supervised,
    )


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def text_config(kind: str, vocab_size: int) -> TextConfig:
    if kind == "main":
        return TextConfig(
            vocab_size=vocab_size,
            seq_len=int(os.environ.get("FLM_V07_MAIN_SEQ", "512")),
            n_layer=int(os.environ.get("FLM_V07_MAIN_LAYERS", "14")),
            n_head=int(os.environ.get("FLM_V07_MAIN_HEADS", "12")),
            n_embd=int(os.environ.get("FLM_V07_MAIN_EMBD", "768")),
        )
    return TextConfig(
        vocab_size=vocab_size,
        seq_len=int(os.environ.get("FLM_V07_CODER_SEQ", "512")),
        n_layer=int(os.environ.get("FLM_V07_CODER_LAYERS", "12")),
        n_head=int(os.environ.get("FLM_V07_CODER_HEADS", "12")),
        n_embd=int(os.environ.get("FLM_V07_CODER_EMBD", "768")),
    )


@torch.no_grad()
def eval_text(model, data, cfg, runtime, rng, batches=8):
    model.eval()
    vals = []
    for _ in range(max(1, batches)):
        starts = sample_starts(rng, len(data), cfg.seq_len, 1)
        x, y = make_xy(data, starts, cfg.seq_len, runtime.device)
        with runtime.autocast():
            _, loss = model(x, y)
            if loss.ndim:
                loss = loss.mean()
        vals.append(float(loss.detach()))
    model.train()
    return sum(vals) / len(vals)


def train_text_pretrain(kind, data, out_root, runtime, vocab_size, steps, batch, accum):
    cfg = text_config(kind, vocab_size)
    base = ByteCausalLM(cfg).to(runtime.device)
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        model = torch.nn.DataParallel(base)
    opt = torch.optim.AdamW(base.parameters(), lr=2.5e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    rng = np.random.default_rng(707 if kind == "main" else 717)
    warmup = max(20, min(800, steps // 20))
    min_lr = 2.5e-5
    start_time = time.time()
    tokens_seen = 0
    last = float("nan")

    for step in range(steps):
        if step < warmup:
            lr = 2.5e-4 * (step + 1) / warmup
        else:
            p = (step - warmup) / max(1, steps - warmup - 1)
            lr = min_lr + 0.5 * (2.5e-4 - min_lr) * (1 + math.cos(math.pi * p))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            starts = sample_starts(rng, len(data), cfg.seq_len, batch)
            x, y = make_xy(data, starts, cfg.seq_len, runtime.device)
            with runtime.autocast():
                _, loss = model(x, y)
                if loss.ndim:
                    loss = loss.mean()
                loss = loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {kind} pretrain loss step={step}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total += float(loss.detach())
            tokens_seen += batch * cfg.seq_len
        if scaler.is_enabled():
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        runtime.optimizer_step(opt, scaler)
        last = total
        if step % 100 == 0 or step == steps - 1:
            print(
                f"v07 {kind} pretrain step={step}/{steps} loss={last:.4f} "
                f"lr={lr:.7f} tokens_seen={tokens_seen}",
                flush=True,
            )

    ev = eval_text(model, data, cfg, runtime, rng, int(os.environ.get("FLM_V07_EVAL_BATCHES", "8")))
    result = {
        "kind": kind,
        "stage": "pretrain",
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "tokens_seen": tokens_seen,
        "train_loss": last,
        "eval_loss": ev,
        "elapsed_s": time.time() - start_time,
        "config": cfg.__dict__,
    }
    stage = out_root / kind
    atomic_torch_save(
        {"model": base.state_dict(), "config": cfg.__dict__, "result": result},
        stage / "checkpoint_pretrain.pt",
    )
    (stage / "metrics_pretrain.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model, opt
    if runtime.kind == "gpu":
        torch.cuda.empty_cache()
    return base, cfg, result


def train_text_sft(kind, base, cfg, data, mask, out_root, runtime, steps, batch, accum):
    model = base
    wrapped = model
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        wrapped = torch.nn.DataParallel(model)
    lr_max = 7e-5 if kind == "main" else 8e-5
    lr_min = 7e-6 if kind == "main" else 8e-6
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    rng = np.random.default_rng(727 if kind == "main" else 737)
    warmup = max(20, min(300, steps // 15))
    start_time = time.time()
    supervised_seen = 0
    last = float("nan")

    for step in range(steps):
        if step < warmup:
            lr = lr_max * (step + 1) / warmup
        else:
            p = (step - warmup) / max(1, steps - warmup - 1)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            x, y, sup = make_sft_batch(data, mask, rng, cfg.seq_len, batch, runtime.device)
            with runtime.autocast():
                _, loss = wrapped(x, y)
                if loss.ndim:
                    loss = loss.mean()
                loss = loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {kind} SFT loss step={step}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total += float(loss.detach())
            supervised_seen += sup
        if scaler.is_enabled():
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        runtime.optimizer_step(opt, scaler)
        last = total
        if step % 100 == 0 or step == steps - 1:
            print(
                f"v07 {kind} sft step={step}/{steps} loss={last:.4f} "
                f"lr={lr:.7f} supervised_seen={supervised_seen}",
                flush=True,
            )

    result = {
        "stage": "sft",
        "steps": steps,
        "supervised_tokens_seen": supervised_seen,
        "train_loss": last,
        "elapsed_s": time.time() - start_time,
    }
    stage = out_root / kind
    atomic_torch_save(
        {"model": model.state_dict(), "config": cfg.__dict__, "result": result},
        stage / "checkpoint.pt",
    )
    (stage / "metrics_sft.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del wrapped, opt
    if runtime.kind == "gpu":
        torch.cuda.empty_cache()
    return model, result


def chat_prefix(tok: Tokenizer, user: str, system: str | None = None) -> list[int]:
    ids = []
    bos = tok.token_to_id("<bos>")
    if bos is not None:
        ids.append(bos)
    if system:
        ids.append(tok.token_to_id("<|system|>"))
        ids.extend(tok.encode(system, add_special_tokens=False).ids)
    ids.append(tok.token_to_id("<|user|>"))
    ids.extend(tok.encode(user, add_special_tokens=False).ids)
    ids.append(tok.token_to_id("<|assistant|>"))
    return ids


@torch.no_grad()
def generate(model, cfg, tok, user, device, *, system=None, max_new=180, temperature=0.35, top_k=24):
    model.eval()
    ids = chat_prefix(tok, user, system)
    end_ids = {x for x in (tok.token_to_id("<|end|>"), tok.token_to_id("<eos>")) if x is not None}
    generated = []
    for _ in range(max_new):
        x = torch.tensor([ids[-cfg.seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        z = logits[0, -1].float() / max(temperature, 1e-5)
        k = min(top_k, z.numel())
        vals, idx = torch.topk(z, k)
        probs = torch.softmax(vals, -1)
        nxt = int(idx[torch.multinomial(probs, 1)].item())
        if nxt in end_ids:
            break
        ids.append(nxt)
        generated.append(nxt)
    return tok.decode(generated, skip_special_tokens=False).strip()


def sample_quality(main, main_cfg, coder, coder_cfg, tok, device):
    main_prompts = [
        "Merhaba! Bana iki cümleyle kendini tanıtır mısın?",
        "Türkiye'nin başkenti neresidir? Kısa cevap ver.",
        "Bir dosyanın SHA-256 özetinin ne işe yaradığını Türkçe açıkla.",
        "Hello! Explain what an operating system does in two sentences.",
    ]
    main_samples = [{"prompt": p, "output": generate(main, main_cfg, tok, p, device)} for p in main_prompts]

    coder_prompts = [
        "İki tamsayıyı toplayan add(a, b) adlı Python fonksiyonu yaz.",
        "Write a Python function is_even(n) that returns whether an integer is even.",
    ]
    coder_samples = [{"prompt": p, "output": generate(coder, coder_cfg, tok, p, device, max_new=240)} for p in coder_prompts]
    tool_system = (
        'Available tools: [{"name":"add","description":"Add two integers",'
        '"parameters":{"type":"object","properties":{"a":{"type":"integer"},'
        '"b":{"type":"integer"}},"required":["a","b"]}}]. '
        'Use <|tool_call|> JSON <|tool_end|> when the tool is required.'
    )
    tool_prompt = "add aracını kullanarak 27 ile 15'i topla."
    tool_output = generate(
        coder, coder_cfg, tok, tool_prompt, device,
        system=tool_system, max_new=180, temperature=0.20, top_k=12,
    )
    return {
        "main": main_samples,
        "coder": coder_samples,
        "tool_call": {"prompt": tool_prompt, "output": tool_output},
    }


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def old_payload(row: dict, op: str) -> str:
    raw = str(row.get("action") or "")
    try:
        tree = ast.parse(raw)
    except Exception:
        return raw[:120] if op == "KEY" else ""
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for call in calls:
        f = call.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if op == "TYPE" and name in {"write", "typewrite"} and call.args:
            x = _literal(call.args[0])
            return str(x) if isinstance(x, str) else ""
        if op == "KEY" and name in {"hotkey", "press"}:
            vals = [_literal(x) for x in call.args]
            return "+".join(str(x) for x in vals if isinstance(x, (str, int)))
        if op == "SCROLL" and name in {"scroll", "hscroll"} and call.args:
            x = _literal(call.args[0])
            if isinstance(x, (int, float)):
                return f"{int(x)},0" if name == "hscroll" else f"0,{int(x)}"
    return raw[:120] if op == "KEY" and len(raw) <= 120 else ""


def stable_eval_key(row: dict) -> str:
    return str(
        row.get("episode_id")
        or row.get("task_id")
        or (str(row.get("archive", "")) + "|" + str(row.get("image", "")))
    )


def heldout(row: dict) -> bool:
    h = hashlib.blake2b(stable_eval_key(row).encode("utf-8", "ignore"), digest_size=2).digest()
    return int.from_bytes(h, "big") % 10 == 0


def merge_cu_rows(v07_root: Path):
    """Load only native v0.7 executable REXX desktop actions.

    v0.7 deliberately does not mix the older v0.5 pseudo/action data. This
    keeps ComputerUse v3 training aligned with the real executable action
    protocol produced by prepare_v07.py.
    """
    rows = []
    histories: dict[str, list[str]] = {}
    for line in (v07_root / "cu07_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["_root"] = "v07"
        episode = str(r.get("episode_id") or "")
        previous = histories.setdefault(episode, [])
        r["_history"] = list(previous[-4:])
        rows.append(r)
        previous.append(
            summarize_action(
                r.get("operation", "OTHER"),
                coord=r.get("coord"),
                coord2=r.get("coord2"),
                payload=r.get("payload", ""),
            )
        )

    train = [r for r in rows if not heldout(r)]
    eval_rows = [r for r in rows if heldout(r)]
    if len(train) < 1000 or len(eval_rows) < 50:
        raise RuntimeError(f"CU v0.7 split too small train={len(train)} eval={len(eval_rows)}")
    return train, eval_rows


class BalancedSampler:
    def __init__(self, rows, seed=747):
        self.rng = random.Random(seed)
        self.by_op = {}
        for r in rows:
            self.by_op.setdefault(r["operation"], []).append(r)
        self.ops = sorted(self.by_op)
        if len(self.ops) < 4:
            raise RuntimeError(f"too few CU action classes: {self.ops}")

    def sample(self, batch):
        out = []
        for _ in range(batch):
            op = self.rng.choice(self.ops)
            out.append(self.rng.choice(self.by_op[op]))
        return out


def payload_pair(text: str, cfg):
    raw = list(str(text).encode("utf-8", "ignore")[: cfg.payload_len - 1])
    inp = [cfg.payload_bos] + raw
    tgt = raw + [cfg.payload_eos]
    inp += [cfg.payload_pad] * (cfg.payload_len - len(inp))
    tgt += [-100] * (cfg.payload_len - len(tgt))
    return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def patch_index(coord, cfg):
    if coord is None:
        return 0
    side = cfg.image_size // cfg.patch
    x = min(0.999999, max(0.0, float(coord[0])))
    y = min(0.999999, max(0.0, float(coord[1])))
    return min(side - 1, int(y * side)) * side + min(side - 1, int(x * side))


def bpe_task(tok, text, cfg, pad_id, history=None):
    ids = build_context_ids(tok, text, history, cfg.task_len, pad_id)
    return torch.tensor(ids, dtype=torch.long)


def cu_batch(rows, stores, tok, cfg, pad_id, domains, device):
    images = []
    tasks = []
    pins = []
    ptgts = []
    ops = []
    p1 = []
    p1v = []
    p2 = []
    p2v = []
    dom = []
    payload_valid = []
    for r in rows:
        store = stores[r["_root"]]
        images.append(store.image(r["archive"], r["image"], cfg.image_size))
        tasks.append(bpe_task(tok, r.get("task", ""), cfg, pad_id, r.get("_history")))
        pi, pt = payload_pair(r.get("payload", ""), cfg)
        pins.append(pi)
        ptgts.append(pt)
        op = str(r["operation"]).upper()
        ops.append(OPS_V3[op])
        coord = r.get("coord")
        coord2 = r.get("coord2")
        p1.append(patch_index(coord, cfg))
        p1v.append(coord is not None and op in {"MOVE", "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "DRAG"})
        p2.append(patch_index(coord2, cfg))
        p2v.append(coord2 is not None and op == "DRAG")
        dom.append(domains[str(r.get("domain") or "")])
        payload_valid.append(op in {"TYPE", "KEY", "KEY_DOWN", "KEY_UP", "SCROLL", "WAIT"})

    return (
        torch.stack(images).to(device),
        torch.stack(tasks).to(device),
        torch.stack(pins).to(device),
        dict(
            op_target=torch.tensor(ops, dtype=torch.long, device=device),
            op_valid=torch.ones(len(rows), dtype=torch.bool, device=device),
            pointer_target=torch.tensor(p1, dtype=torch.long, device=device),
            pointer_valid=torch.tensor(p1v, dtype=torch.bool, device=device),
            pointer2_target=torch.tensor(p2, dtype=torch.long, device=device),
            pointer2_valid=torch.tensor(p2v, dtype=torch.bool, device=device),
            domain_target=torch.tensor(dom, dtype=torch.long, device=device),
            domain_valid=torch.ones(len(rows), dtype=torch.bool, device=device),
            payload_target=torch.stack(ptgts).to(device),
            payload_valid=torch.tensor(payload_valid, dtype=torch.bool, device=device),
        ),
    )


@torch.no_grad()
def eval_cu(model, rows, stores, tok, cfg, pad_id, domains, device, batches=64):
    model.eval()
    rng = random.Random(757)
    op_ok = op_n = ptr_n = ptr_hit = 0
    ptr_l2 = []
    side = cfg.image_size // cfg.patch
    for _ in range(min(batches, len(rows))):
        r = rng.choice(rows)
        image, task, pin, kw = cu_batch([r], stores, tok, cfg, pad_id, domains, device)
        out, loss, _ = model(image, task, pin, **kw)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite CU eval loss")
        op_n += 1
        op_ok += int(int(out["op_logits"].argmax(-1).item()) == int(kw["op_target"].item()))
        if bool(kw["pointer_valid"].item()):
            ptr_n += 1
            truth = r["coord"]
            pred = out["coord"][0].float().cpu().tolist()
            d = math.hypot(pred[0] - truth[0], pred[1] - truth[1])
            ptr_l2.append(d)
            ptr_hit += int(d <= max(0.10, 1.5 / side))
    model.train()
    return {
        "heldout_examples": len(rows),
        "op_accuracy": op_ok / max(1, op_n),
        "pointer_examples": ptr_n,
        "pointer_hit_0p1": ptr_hit / max(1, ptr_n),
        "pointer_mean_l2": sum(ptr_l2) / max(1, len(ptr_l2)),
    }


def train_computer_use(v07_root, out_root, runtime, tok, vocab_size, steps):
    train_rows, eval_rows = merge_cu_rows(v07_root)
    domain_names = sorted({str(r.get("domain") or "") for r in train_rows + eval_rows})
    if len(domain_names) >= 256:
        raise RuntimeError(f"too many CU domains: {len(domain_names)}")
    domains = {name: i + 1 for i, name in enumerate(domain_names)}
    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        raise RuntimeError("tokenizer missing <pad>")

    cfg = ComputerUseV3Config(
        text_vocab=vocab_size,
        task_pad_token=pad_id,
        task_len=int(os.environ.get("FLM_V07_CU_TASK_LEN", "160")),
        image_size=int(os.environ.get("FLM_V07_CU_IMAGE", "224")),
        embd=int(os.environ.get("FLM_V07_CU_EMBD", "768")),
        text_layers=int(os.environ.get("FLM_V07_CU_TEXT_LAYERS", "4")),
        vision_layers=int(os.environ.get("FLM_V07_CU_VISION_LAYERS", "8")),
        n_head=int(os.environ.get("FLM_V07_CU_HEADS", "12")),
        num_domains=256,
        payload_len=int(os.environ.get("FLM_V07_CU_PAYLOAD_LEN", "128")),
    )
    batch = int(os.environ.get("FLM_V07_CU_BATCH", "4"))
    stores = {"v07": ZipImageStore(v07_root)}
    sampler = BalancedSampler(train_rows)

    # v0.6 fp16 eventually produced a non-finite loss. v0.7 deliberately keeps
    # this ~100M multimodal policy in FP32 and lowers the learning rate.
    base = ComputerUseV3(cfg).to(runtime.device)
    wrapped = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        wrapped = torch.nn.DataParallel(base)
    opt = torch.optim.AdamW(base.parameters(), lr=8e-5, betas=(0.9, 0.95), weight_decay=0.05)
    start = time.time()
    last = float("nan")
    action_counts = {op: len(rows) for op, rows in sampler.by_op.items()}

    for step in range(steps):
        picked = sampler.sample(batch)
        image, task, pin, kw = cu_batch(picked, stores, tok, cfg, pad_id, domains, runtime.device)
        if not torch.isfinite(image).all():
            raise RuntimeError(f"non-finite CU image input at step {step}")
        opt.zero_grad(set_to_none=True)
        out, loss, parts = wrapped(image, task, pin, **kw)
        if loss.ndim:
            loss = loss.mean()
        if not torch.isfinite(loss):
            diag = {
                "step": step,
                "rows": [{k: r.get(k) for k in ("source", "domain", "operation", "archive", "image")} for r in picked],
                "parts": {k: float(v.detach().float().mean()) for k, v in parts.items()},
            }
            (out_root / "computer_use_nonfinite.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
            raise RuntimeError(f"non-finite CU v0.7 loss at step {step}: {diag}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(base.parameters(), 0.5)
        if not torch.isfinite(torch.as_tensor(grad)):
            raise RuntimeError(f"non-finite CU gradient at step {step}")
        opt.step()
        last = float(loss.detach())
        if step % 100 == 0 or step == steps - 1:
            pv = {k: float(v.detach().float().mean()) for k, v in parts.items()}
            print(f"v07 computer_use step={step}/{steps} loss={last:.4f} parts={pv}", flush=True)
        if step > 0 and step % 2000 == 0:
            atomic_torch_save(
                {"model": base.state_dict(), "config": cfg.__dict__, "domains": domains, "step": step},
                out_root / "computer_use" / "checkpoint_resume.pt",
            )

    ev = eval_cu(
        base, eval_rows, stores, tok, cfg, pad_id, domains, runtime.device,
        int(os.environ.get("FLM_V07_CU_EVAL_EXAMPLES", "96")),
    )
    result = {
        "kind": "computer_use",
        "stage": "v3-pointer-policy",
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "train_loss": last,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "domains": len(domain_names),
        "action_counts": action_counts,
        **ev,
        "elapsed_s": time.time() - start,
        "config": cfg.__dict__,
    }
    out = out_root / "computer_use"
    atomic_torch_save(
        {"model": base.state_dict(), "config": cfg.__dict__, "domains": domains, "result": result},
        out / "checkpoint.pt",
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("V07_CU_RESULT=" + json.dumps(result), flush=True)
    return result


def checkpoint_result(out_root: Path, summary: dict):
    (out_root / "suite_progress.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("all", "main", "coder", "computer_use"), default="all")
    args = ap.parse_args()

    v07 = resolve_pipeline_root("v0.7")
    out = Path(os.environ.get("FLM_V07_OUTPUT_ROOT", "/kaggle/working/flm-v0.7-full"))
    out.mkdir(parents=True, exist_ok=True)
    runtime = select_runtime("gpu")
    random.seed(707)
    torch.manual_seed(707)
    if runtime.kind == "gpu":
        torch.cuda.manual_seed_all(707)

    tok = Tokenizer.from_file(str(v07 / "tokenizer.json"))
    vocab_size = tok.get_vocab_size()
    if vocab_size < 8192:
        raise RuntimeError(f"v0.7 tokenizer too small: {vocab_size}")

    batch = int(os.environ.get("FLM_V07_TEXT_BATCH", "8"))
    accum = int(os.environ.get("FLM_V07_TEXT_ACCUM", "2"))
    main_steps = int(os.environ.get("FLM_V07_MAIN_STEPS", "25000"))
    main_sft_steps = int(os.environ.get("FLM_V07_MAIN_SFT_STEPS", "3500"))
    coder_steps = int(os.environ.get("FLM_V07_CODER_STEPS", "15000"))
    coder_sft_steps = int(os.environ.get("FLM_V07_CODER_SFT_STEPS", "3500"))
    cu_steps = int(os.environ.get("FLM_V07_CU_STEPS", "8000"))

    summary = {
        "pipeline_version": "v0.7",
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "tokenizer_vocab_size": vocab_size,
        "models": {},
    }

    main_model = main_cfg = coder_model = coder_cfg = None

    if args.only in {"all", "main"}:
        main_data = mmap_tokens(v07 / "main_train.u16")
        main_sft_data = mmap_tokens(v07 / "main_sft_tokens.u16")
        main_sft_mask = mmap_mask(v07 / "main_sft_mask.u8")
        if len(main_sft_data) != len(main_sft_mask):
            raise RuntimeError("main SFT token/mask mismatch")
        main_model, main_cfg, pre = train_text_pretrain(
            "main", main_data, out, runtime, vocab_size, main_steps, batch, accum
        )
        main_model, sft = train_text_sft(
            "main", main_model, main_cfg, main_sft_data, main_sft_mask,
            out, runtime, main_sft_steps, batch, accum,
        )
        summary["models"]["main"] = {"pretrain": pre, "sft": sft}
        checkpoint_result(out, summary)

    if args.only in {"all", "coder"}:
        coder_data = mmap_tokens(v07 / "coder_train.u16")
        coder_sft_data = mmap_tokens(v07 / "coder_sft_tokens.u16")
        coder_sft_mask = mmap_mask(v07 / "coder_sft_mask.u8")
        if len(coder_sft_data) != len(coder_sft_mask):
            raise RuntimeError("coder SFT token/mask mismatch")
        coder_model, coder_cfg, pre = train_text_pretrain(
            "coder", coder_data, out, runtime, vocab_size, coder_steps, batch, accum
        )
        coder_model, sft = train_text_sft(
            "coder", coder_model, coder_cfg, coder_sft_data, coder_sft_mask,
            out, runtime, coder_sft_steps, batch, accum,
        )
        summary["models"]["coder"] = {"pretrain": pre, "sft": sft}
        checkpoint_result(out, summary)

    if args.only == "all" and main_model is not None and coder_model is not None:
        quality = sample_quality(main_model, main_cfg, coder_model, coder_cfg, tok, runtime.device)
        summary["quality_samples"] = quality
        (out / "quality_samples.json").write_text(
            json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("V07_QUALITY_SAMPLES=" + json.dumps(quality, ensure_ascii=False), flush=True)
        del main_model, coder_model
        if runtime.kind == "gpu":
            torch.cuda.empty_cache()

    if args.only in {"all", "computer_use"}:
        cu = train_computer_use(v07, out, runtime, tok, vocab_size, cu_steps)
        summary["models"]["computer_use"] = cu
        checkpoint_result(out, summary)

    for name in ("tokenizer.json", "tokenizer_meta.json", "sources.json"):
        shutil.copy2(v07 / name, out / ("text_sources.json" if name == "sources.json" else name))
    (out / "suite_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"V07_SUITE_COMPLETE output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
