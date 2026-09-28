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
from flm.quality_v07 import validate_suite
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
    env_name = "FLM_V07_DATA_ROOT" if version.startswith("v0.7") else "FLM_V05_DATA_ROOT"
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


def validate_pipeline_inputs(text_root: Path, computer_root: Path) -> dict:
    text_manifest = json.loads((text_root / "sources.json").read_text(encoding="utf-8"))
    computer_manifest = json.loads((computer_root / "sources.json").read_text(encoding="utf-8"))

    if text_manifest.get("pipeline_version") != "v0.7-dev-text":
        raise RuntimeError(f"wrong text pipeline: {text_manifest.get('pipeline_version')}")
    if computer_manifest.get("pipeline_version") != "v0.7-dev-computer":
        raise RuntimeError(f"wrong ComputerUse pipeline: {computer_manifest.get('pipeline_version')}")
    if int(text_manifest.get("data_revision", 0)) < 2:
        raise RuntimeError("v0.7 text data revision 2+ is required")
    if int(computer_manifest.get("data_revision", 0)) < 2:
        raise RuntimeError("v0.7 ComputerUse data revision 2+ is required")

    ts = text_manifest.get("stats") or {}
    cs = computer_manifest.get("stats") or {}
    required_text = {
        "main_tokens": int((ts.get("main") or {}).get("tokens", 0)),
        "coder_tokens": int((ts.get("coder") or {}).get("tokens", 0)),
        "main_sft_supervised": int((ts.get("main_sft") or {}).get("supervised_tokens", 0)),
        "coder_sft_supervised": int((ts.get("coder_sft") or {}).get("supervised_tokens", 0)),
        "tokenizer_vocab": int((ts.get("tokenizer") or {}).get("vocab_size", 0)),
    }
    if required_text["main_tokens"] < 1_000_000_000:
        raise RuntimeError(f"main token corpus too small: {required_text}")
    if required_text["coder_tokens"] < 190_000_000:
        raise RuntimeError(f"coder token corpus too small: {required_text}")
    if required_text["main_sft_supervised"] < 300_000_000:
        raise RuntimeError(f"main SFT corpus too small: {required_text}")
    if required_text["coder_sft_supervised"] < 90_000_000:
        raise RuntimeError(f"coder SFT corpus too small: {required_text}")
    if required_text["tokenizer_vocab"] < 30_000:
        raise RuntimeError(f"tokenizer too small: {required_text}")

    examples = int(cs.get("examples", 0))
    ops = cs.get("ops") or {}
    sources = cs.get("sources") or {}
    domains = int(cs.get("domains", 0))
    if examples < 30_000:
        raise RuntimeError(f"ComputerUse corpus too small: {examples}")
    minimum_ops = {
        "CLICK": 20_000, "KEY": 1_000, "TYPE": 1_000, "SCROLL": 750,
        "DRAG": 300, "RIGHT_CLICK": 200, "DOUBLE_CLICK": 200, "DONE": 500,
    }
    short = {k: (int(ops.get(k, 0)), v) for k, v in minimum_ops.items() if int(ops.get(k, 0)) < v}
    if short:
        raise RuntimeError(f"ComputerUse multi-action coverage too small: {short}; all={ops}")
    if len(sources) < 2 or domains < 2:
        raise RuntimeError(f"ComputerUse source/domain diversity too small: sources={sources} domains={domains}")

    result = {
        "text": required_text,
        "computer": {
            "examples": examples,
            "ops": ops,
            "sources": sources,
            "domains": domains,
        },
    }
    print("V07_INPUT_QUALITY_OK=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def mmap_tokens(path: Path) -> np.memmap:
    if not path.is_file() or path.stat().st_size < 4096:
        raise RuntimeError(f"token file missing/small: {path}")
    return np.memmap(path, mode="r", dtype="<u2")


def mmap_mask(path: Path) -> np.memmap:
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"mask file missing/small: {path}")
    return np.memmap(path, mode="r", dtype=np.uint8)


def split_train_eval_stream(data, seq_len: int, *, eval_fraction: float = 0.01):
    n = len(data)
    minimum = seq_len * 32
    if n < minimum * 3:
        raise RuntimeError(f"token stream too small for holdout: n={n} seq={seq_len}")
    holdout = max(minimum, int(n * eval_fraction))
    holdout = min(holdout, n // 5)
    cut = n - holdout
    if cut <= seq_len * 4 or holdout <= seq_len * 4:
        raise RuntimeError(f"invalid holdout split: n={n} cut={cut} holdout={holdout}")
    return data[:cut], data[cut:]


class ShuffledStartPool:
    """Epoch-aware start sampler: every block is used once before reshuffling."""
    def __init__(self, starts: np.ndarray, seed: int):
        starts = np.asarray(starts, dtype=np.int64)
        if starts.size == 0:
            raise RuntimeError("empty shuffled start pool")
        self.starts = starts
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(starts))
        self.pos = 0
        self.epoch = 0

    def take(self, count: int) -> np.ndarray:
        parts = []
        need = int(count)
        while need > 0:
            remaining = len(self.order) - self.pos
            if remaining == 0:
                self.epoch += 1
                self.order = self.rng.permutation(len(self.starts))
                self.pos = 0
                remaining = len(self.order)
            n = min(need, remaining)
            idx = self.order[self.pos:self.pos + n]
            parts.append(self.starts[idx])
            self.pos += n
            need -= n
        return np.concatenate(parts)

    def state_dict(self) -> dict:
        return {
            "order": self.order.copy(),
            "pos": int(self.pos),
            "epoch": int(self.epoch),
            "rng_state": self.rng.bit_generator.state,
            "starts_len": int(len(self.starts)),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("starts_len", -1)) != len(self.starts):
            raise RuntimeError(
                f"sampler dataset changed: checkpoint={state.get('starts_len')} current={len(self.starts)}"
            )
        order = np.asarray(state["order"], dtype=np.int64)
        if len(order) != len(self.starts):
            raise RuntimeError("invalid sampler order length")
        self.order = order
        self.pos = int(state["pos"])
        self.epoch = int(state["epoch"])
        self.rng.bit_generator.state = state["rng_state"]


def nonoverlap_starts(length: int, seq: int) -> np.ndarray:
    high = length - seq - 1
    if high <= 0:
        raise RuntimeError(f"dataset too short for seq={seq}: {length}")
    # seq+1 keeps x/y windows disjoint, so an epoch has real coverage semantics.
    starts = np.arange(0, high, seq + 1, dtype=np.int64)
    if starts.size == 0:
        raise RuntimeError(f"no training blocks for seq={seq}: {length}")
    return starts


def eligible_sft_starts(mask, seq: int, min_supervised: int = 8) -> np.ndarray:
    """Build non-overlapping SFT windows that contain useful assistant targets."""
    stride = seq + 1
    total = max(0, (len(mask) - 1) // stride)
    if total <= 0:
        raise RuntimeError("SFT mask too short")

    accepted = []
    chunk_blocks = 32768
    for block0 in range(0, total, chunk_blocks):
        n = min(chunk_blocks, total - block0)
        start = block0 * stride
        stop = start + n * stride
        chunk = np.asarray(mask[start:stop], dtype=np.uint8).reshape(n, stride)
        # y targets use positions s+1 .. s+seq.
        counts = chunk[:, 1:seq + 1].sum(axis=1)
        good = np.flatnonzero(counts >= min_supervised)
        if good.size:
            accepted.append((block0 + good).astype(np.int64) * stride)

    if not accepted:
        raise RuntimeError("no SFT blocks contain enough supervised targets")
    return np.concatenate(accepted)


def make_sft_batch_at_starts(data, mask, starts, seq, device):
    xs, ys = [], []
    supervised = 0
    for s in np.asarray(starts, dtype=np.int64):
        s = int(s)
        target_mask = np.asarray(mask[s + 1:s + seq + 1], dtype=np.bool_)
        x = np.asarray(data[s:s + seq], dtype=np.int64)
        y = np.asarray(data[s + 1:s + seq + 1], dtype=np.int64).copy()
        y[~target_mask] = -100
        xs.append(x)
        ys.append(y)
        supervised += int(target_mask.sum())
    return (
        torch.from_numpy(np.stack(xs)).to(device),
        torch.from_numpy(np.stack(ys)).to(device),
        supervised,
    )


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


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def resume_enabled() -> bool:
    return os.environ.get("FLM_V07_RESUME", "1").strip().lower() not in {"0", "false", "no", "off"}


def text_config(kind: str, vocab_size: int) -> TextConfig:
    pos = os.environ.get("FLM_V07_POSITION_ENCODING", "rope")
    if kind == "main":
        return TextConfig(
            vocab_size=vocab_size,
            seq_len=int(os.environ.get("FLM_V07_MAIN_SEQ", "4096")),
            n_layer=int(os.environ.get("FLM_V07_MAIN_LAYERS", "14")),
            n_head=int(os.environ.get("FLM_V07_MAIN_HEADS", "12")),
            n_embd=int(os.environ.get("FLM_V07_MAIN_EMBD", "768")),
            position_encoding=pos,
        )
    return TextConfig(
        vocab_size=vocab_size,
        seq_len=int(os.environ.get("FLM_V07_CODER_SEQ", "4096")),
        n_layer=int(os.environ.get("FLM_V07_CODER_LAYERS", "14")),
        n_head=int(os.environ.get("FLM_V07_CODER_HEADS", "12")),
        n_embd=int(os.environ.get("FLM_V07_CODER_EMBD", "768")),
        position_encoding=pos,
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


@torch.no_grad()
def eval_sft(model, data, mask, cfg, runtime, rng, batches=8):
    model.eval()
    vals = []
    supervised = 0
    for _ in range(max(1, batches)):
        x, y, sup = make_sft_batch(
            data, mask, rng, cfg.seq_len, 1, runtime.device, min_supervised=4
        )
        with runtime.autocast():
            _, loss = model(x, y)
            if loss.ndim:
                loss = loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite SFT eval loss")
        vals.append(float(loss.detach()))
        supervised += int(sup)
    model.train()
    return sum(vals) / len(vals), supervised


def train_text_pretrain(kind, data, out_root, runtime, vocab_size, steps, batch, accum, init_checkpoint=None):
    cfg = text_config(kind, vocab_size)
    train_data, eval_data = split_train_eval_stream(data, cfg.seq_len, eval_fraction=0.01)
    base = ByteCausalLM(cfg).to(runtime.device)
    if init_checkpoint is not None:
        init_checkpoint = Path(init_checkpoint)
        if not init_checkpoint.is_file():
            raise RuntimeError(f"missing initialization checkpoint: {init_checkpoint}")
        seed = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        seed_cfg = TextConfig(**seed["config"])
        if seed_cfg != cfg:
            raise RuntimeError(
                f"{kind} Main-inheritance config mismatch: main={seed_cfg} target={cfg}"
            )
        base.load_state_dict(seed["model"], strict=True)
        print(f"V07_{kind.upper()}_INITIALIZED_FROM_MAIN={init_checkpoint}", flush=True)
    base.gradient_checkpointing = os.environ.get(
        "FLM_V07_GRADIENT_CHECKPOINTING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if base.gradient_checkpointing:
        print(f"V07_{kind.upper()}_ACTIVATION_CHECKPOINTING=1", flush=True)
    model = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        model = torch.nn.DataParallel(base)

    if kind == "coder" and init_checkpoint is not None:
        lr_max = float(os.environ.get("FLM_V07_CODER_PRETRAIN_LR", "1.2e-4"))
        min_lr = float(os.environ.get("FLM_V07_CODER_PRETRAIN_MIN_LR", "1.2e-5"))
    else:
        lr_max = float(os.environ.get("FLM_V07_PRETRAIN_LR", "2.5e-4"))
        min_lr = float(os.environ.get("FLM_V07_PRETRAIN_MIN_LR", "2.5e-5"))
    opt = torch.optim.AdamW(base.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    rng = np.random.default_rng(707 if kind == "main" else 717)
    eval_rng = np.random.default_rng(1707 if kind == "main" else 1717)
    train_starts = nonoverlap_starts(len(train_data), cfg.seq_len)
    train_pool = ShuffledStartPool(
        train_starts, 2707 if kind == "main" else 2717
    )
    windows_per_step = batch * accum
    steps_per_epoch = math.ceil(len(train_starts) / max(1, windows_per_step))
    requested_steps = steps
    target_epochs = float(os.environ.get(
        f"FLM_V07_{kind.upper()}_PRETRAIN_EPOCHS", "1.0"
    ))
    if steps <= 0:
        steps = max(1, math.ceil(steps_per_epoch * target_epochs))
    print(
        f"V07_{kind.upper()}_COVERAGE blocks={len(train_starts)} "
        f"windows_per_step={windows_per_step} steps_per_epoch={steps_per_epoch} "
        f"requested_steps={requested_steps} target_epochs={target_epochs:.3f} "
        f"configured_steps={steps}",
        flush=True,
    )
    warmup = max(20, min(800, steps // 20))
    eval_interval = int(os.environ.get("FLM_V07_TEXT_EVAL_INTERVAL", str(max(250, min(2000, max(1, steps // 12))))))
    checkpoint_interval = int(os.environ.get("FLM_V07_TEXT_CHECKPOINT_INTERVAL", "5000"))
    eval_batches = int(os.environ.get("FLM_V07_EVAL_BATCHES", "12"))
    start_time = time.time()
    tokens_seen = 0
    last = float("nan")
    best_eval = float("inf")
    best_step = -1
    stage = out_root / kind
    best_path = stage / "checkpoint_best_pretrain.pt"
    resume_path = stage / "checkpoint_resume_pretrain.pt"
    final_path = stage / "checkpoint_pretrain.pt"
    start_step = 0

    if resume_enabled() and final_path.is_file():
        done = torch.load(final_path, map_location=runtime.device, weights_only=False)
        done_result = done.get("result") or {}
        if int(done_result.get("steps", 0)) >= steps and done.get("config") == cfg.__dict__:
            base.load_state_dict(done["model"])
            print(f"V07_{kind.upper()}_PRETRAIN_ALREADY_COMPLETE steps={done_result.get('steps')}", flush=True)
            return base, cfg, done_result

    if resume_enabled() and resume_path.is_file():
        ck = torch.load(resume_path, map_location=runtime.device, weights_only=False)
        required = {"model", "optimizer", "sampler", "next_step"}
        if required.issubset(ck):
            if ck.get("config") != cfg.__dict__:
                raise RuntimeError(f"{kind} pretrain resume config mismatch")
            base.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            optimizer_to_device(opt, runtime.device)
            if scaler.is_enabled() and ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            train_pool.load_state_dict(ck["sampler"])
            start_step = int(ck["next_step"])
            tokens_seen = int(ck.get("tokens_seen", 0))
            last = float(ck.get("train_loss", float("nan")))
            best_eval = float(ck.get("best_eval_loss", float("inf")))
            best_step = int(ck.get("best_step", -1))
            print(
                f"V07_{kind.upper()}_PRETRAIN_RESUME step={start_step}/{steps} "
                f"tokens_seen={tokens_seen} sampler_epoch={train_pool.epoch}",
                flush=True,
            )

    for step in range(start_step, steps):
        if step < warmup:
            lr = lr_max * (step + 1) / warmup
        else:
            p = (step - warmup) / max(1, steps - warmup - 1)
            lr = min_lr + 0.5 * (lr_max - min_lr) * (1 + math.cos(math.pi * p))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            starts = train_pool.take(batch)
            x, y = make_xy(train_data, starts, cfg.seq_len, runtime.device)
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
        grad = torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(grad)):
            raise RuntimeError(f"non-finite {kind} pretrain gradient step={step}")
        runtime.optimizer_step(opt, scaler)
        last = total

        should_eval = (step + 1) % eval_interval == 0 or step == steps - 1
        if should_eval:
            ev_now = eval_text(model, eval_data, cfg, runtime, eval_rng, eval_batches)
            print(
                f"v07 {kind} pretrain eval step={step + 1}/{steps} "
                f"eval_loss={ev_now:.4f} best={best_eval:.4f}",
                flush=True,
            )
            if math.isfinite(ev_now) and ev_now < best_eval:
                best_eval = ev_now
                best_step = step + 1
                atomic_torch_save(
                    {
                        "model": base.state_dict(),
                        "config": cfg.__dict__,
                        "step": best_step,
                        "eval_loss": best_eval,
                        "tokens_seen": tokens_seen,
                    },
                    best_path,
                )

        if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            atomic_torch_save(
                {
                    "model": base.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "sampler": train_pool.state_dict(),
                    "config": cfg.__dict__,
                    "step": step + 1,
                    "next_step": step + 1,
                    "tokens_seen": tokens_seen,
                    "train_loss": last,
                    "best_eval_loss": best_eval,
                    "best_step": best_step,
                },
                resume_path,
            )

        if step % 100 == 0 or step == steps - 1:
            print(
                f"v07 {kind} pretrain step={step}/{steps} loss={last:.4f} "
                f"lr={lr:.7f} tokens_seen={tokens_seen}",
                flush=True,
            )

    if best_path.is_file():
        best_obj = torch.load(best_path, map_location=runtime.device, weights_only=False)
        base.load_state_dict(best_obj["model"])
    final_eval = eval_text(base, eval_data, cfg, runtime, eval_rng, eval_batches)
    result = {
        "kind": kind,
        "stage": "pretrain",
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "parameters": sum(p.numel() for p in base.parameters()),
        "steps": steps,
        "tokens_seen": tokens_seen,
        "train_tokens": len(train_data),
        "heldout_tokens": len(eval_data),
        "nonoverlap_blocks": len(train_starts),
        "steps_per_epoch": steps_per_epoch,
        "completed_epochs": tokens_seen / max(1, len(train_starts) * cfg.seq_len),
        "sampler": "shuffled_nonoverlap_no_replacement",
        "train_loss": last,
        "eval_loss": final_eval,
        "best_eval_loss": best_eval,
        "best_step": best_step,
        "elapsed_s": time.time() - start_time,
        "config": cfg.__dict__,
    }
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
    if len(data) != len(mask):
        raise RuntimeError(f"{kind} SFT token/mask mismatch")
    train_data, eval_data = split_train_eval_stream(data, cfg.seq_len, eval_fraction=0.02)
    cut = len(train_data)
    train_mask = mask[:cut]
    eval_mask = mask[cut:]

    model = base
    wrapped = model
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        wrapped = torch.nn.DataParallel(model)
    lr_max = 7e-5 if kind == "main" else 8e-5
    lr_min = 7e-6 if kind == "main" else 8e-6
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.kind == "gpu")
    rng = np.random.default_rng(727 if kind == "main" else 737)
    eval_rng = np.random.default_rng(1727 if kind == "main" else 1737)
    sft_starts = eligible_sft_starts(train_mask, cfg.seq_len, min_supervised=8)
    sft_pool = ShuffledStartPool(
        sft_starts, 3727 if kind == "main" else 3737
    )
    sft_windows_per_step = batch * accum
    sft_steps_per_epoch = math.ceil(
        len(sft_starts) / max(1, sft_windows_per_step)
    )
    requested_steps = steps
    default_epochs = "1.25" if kind == "coder" else "1.0"
    target_epochs = float(os.environ.get(
        f"FLM_V07_{kind.upper()}_SFT_EPOCHS", default_epochs
    ))
    if steps <= 0:
        steps = max(1, math.ceil(sft_steps_per_epoch * target_epochs))
    print(
        f"V07_{kind.upper()}_SFT_COVERAGE blocks={len(sft_starts)} "
        f"windows_per_step={sft_windows_per_step} "
        f"steps_per_epoch={sft_steps_per_epoch} requested_steps={requested_steps} "
        f"target_epochs={target_epochs:.3f} configured_steps={steps}",
        flush=True,
    )
    warmup = max(20, min(300, steps // 15))
    eval_interval = int(os.environ.get("FLM_V07_SFT_EVAL_INTERVAL", str(max(200, min(1000, max(1, steps // 10))))))
    checkpoint_interval = int(os.environ.get("FLM_V07_TEXT_CHECKPOINT_INTERVAL", "5000"))
    eval_batches = int(os.environ.get("FLM_V07_EVAL_BATCHES", "12"))
    start_time = time.time()
    supervised_seen = 0
    last = float("nan")
    best_eval = float("inf")
    best_step = -1
    stage = out_root / kind
    best_path = stage / "checkpoint_best_sft.pt"
    resume_path = stage / "checkpoint_resume_sft.pt"
    final_path = stage / "checkpoint.pt"
    start_step = 0

    if resume_enabled() and final_path.is_file():
        done = torch.load(final_path, map_location=runtime.device, weights_only=False)
        done_result = done.get("result") or {}
        if int(done_result.get("steps", 0)) >= steps and done.get("config") == cfg.__dict__:
            model.load_state_dict(done["model"])
            print(f"V07_{kind.upper()}_SFT_ALREADY_COMPLETE steps={done_result.get('steps')}", flush=True)
            return model, done_result

    if resume_enabled() and resume_path.is_file():
        ck = torch.load(resume_path, map_location=runtime.device, weights_only=False)
        required = {"model", "optimizer", "sampler", "next_step"}
        if required.issubset(ck):
            if ck.get("config") != cfg.__dict__:
                raise RuntimeError(f"{kind} SFT resume config mismatch")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            optimizer_to_device(opt, runtime.device)
            if scaler.is_enabled() and ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            sft_pool.load_state_dict(ck["sampler"])
            start_step = int(ck["next_step"])
            supervised_seen = int(ck.get("supervised_seen", 0))
            last = float(ck.get("train_loss", float("nan")))
            best_eval = float(ck.get("best_eval_loss", float("inf")))
            best_step = int(ck.get("best_step", -1))
            print(
                f"V07_{kind.upper()}_SFT_RESUME step={start_step}/{steps} "
                f"supervised_seen={supervised_seen} sampler_epoch={sft_pool.epoch}",
                flush=True,
            )

    for step in range(start_step, steps):
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
            starts = sft_pool.take(batch)
            x, y, sup = make_sft_batch_at_starts(
                train_data, train_mask, starts, cfg.seq_len, runtime.device
            )
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
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(grad)):
            raise RuntimeError(f"non-finite {kind} SFT gradient step={step}")
        runtime.optimizer_step(opt, scaler)
        last = total

        should_eval = (step + 1) % eval_interval == 0 or step == steps - 1
        if should_eval:
            ev_now, ev_sup = eval_sft(
                wrapped, eval_data, eval_mask, cfg, runtime, eval_rng, eval_batches
            )
            print(
                f"v07 {kind} sft eval step={step + 1}/{steps} "
                f"eval_loss={ev_now:.4f} supervised={ev_sup} best={best_eval:.4f}",
                flush=True,
            )
            if math.isfinite(ev_now) and ev_now < best_eval:
                best_eval = ev_now
                best_step = step + 1
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.__dict__,
                        "step": best_step,
                        "eval_loss": best_eval,
                        "supervised_seen": supervised_seen,
                    },
                    best_path,
                )

        if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "sampler": sft_pool.state_dict(),
                    "config": cfg.__dict__,
                    "step": step + 1,
                    "next_step": step + 1,
                    "supervised_seen": supervised_seen,
                    "train_loss": last,
                    "best_eval_loss": best_eval,
                    "best_step": best_step,
                },
                resume_path,
            )

        if step % 100 == 0 or step == steps - 1:
            print(
                f"v07 {kind} sft step={step}/{steps} loss={last:.4f} "
                f"lr={lr:.7f} supervised_seen={supervised_seen}",
                flush=True,
            )

    if best_path.is_file():
        best_obj = torch.load(best_path, map_location=runtime.device, weights_only=False)
        model.load_state_dict(best_obj["model"])
    final_eval, eval_supervised = eval_sft(
        model, eval_data, eval_mask, cfg, runtime, eval_rng, eval_batches
    )
    result = {
        "stage": "sft",
        "steps": steps,
        "supervised_tokens_seen": supervised_seen,
        "heldout_tokens": len(eval_data),
        "heldout_supervised_sampled": eval_supervised,
        "eligible_nonoverlap_blocks": len(sft_starts),
        "steps_per_epoch": sft_steps_per_epoch,
        "sampler": "shuffled_supervised_nonoverlap_no_replacement",
        "train_loss": last,
        "eval_loss": final_eval,
        "best_eval_loss": best_eval,
        "best_step": best_step,
        "elapsed_s": time.time() - start_time,
    }
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
        "Fotosentez sırasında bitkiler ışık enerjisini nasıl kullanır? Kısa ve bilimsel açıkla.",
        "Hello! Explain what an operating system does in two sentences.",
        "What is the practical difference between RAM and persistent storage? Answer briefly.",
    ]
    main_samples = [{"prompt": p, "output": generate(main, main_cfg, tok, p, device)} for p in main_prompts]

    coder_prompts = [
        "İki tamsayıyı toplayan add(a, b) adlı Python fonksiyonu yaz.",
        "Write a Python function is_even(n) that returns whether an integer is even.",
    ]
    coder_samples = [{"prompt": p, "output": generate(coder, coder_cfg, tok, p, device, max_new=320)} for p in coder_prompts]

    normal_prompt = (
        "Dependency injection nedir? Tool kullanmadan, plan etiketi yazmadan iki kısa paragrafla açıkla."
    )
    normal_output = generate(
        coder, coder_cfg, tok, normal_prompt, device,
        max_new=320, temperature=0.25, top_k=20,
    )
    planning_system = (
        "You are FLM-Coder. For complex software-engineering tasks, first emit "
        "<|plan|> with a concrete multi-step implementation/debugging plan, close it "
        "with <|plan_end|>, then provide the usable answer under <|final|>. "
        "Do not invent tool results."
    )
    planning_prompt = (
        "Bir Python servisinde testler yalnız production yapılandırmasında başarısız oluyor. "
        "Sorunu güvenli biçimde teşhis edip düzeltmek için adım adım plan oluştur; "
        "logları, yapılandırma farklarını, minimal reproducer'ı, testleri ve rollback'i kapsa."
    )
    planning_output = generate(
        coder, coder_cfg, tok, planning_prompt, device,
        system=planning_system, max_new=640, temperature=0.25, top_k=20,
    )
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
        "coder_normal": {"prompt": normal_prompt, "output": normal_output},
        "coder_planning": {"prompt": planning_prompt, "output": planning_output},
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
        r["_history"] = list(previous[-8:])
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

    def state_dict(self) -> dict:
        return {"rng_state": self.rng.getstate(), "ops": list(self.ops)}

    def load_state_dict(self, state: dict) -> None:
        if list(state.get("ops") or []) != self.ops:
            raise RuntimeError("ComputerUse action classes changed since checkpoint")
        self.rng.setstate(state["rng_state"])


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
    op_by_class: dict[str, dict[str, int]] = {}
    ptr_l2 = []
    side = cfg.image_size // cfg.patch
    for _ in range(min(batches, len(rows))):
        r = rng.choice(rows)
        image, task, pin, kw = cu_batch([r], stores, tok, cfg, pad_id, domains, device)
        out, loss, _ = model(image, task, pin, **kw)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite CU eval loss")
        op_n += 1
        pred_op = int(out["op_logits"].argmax(-1).item())
        true_op = int(kw["op_target"].item())
        correct = int(pred_op == true_op)
        op_ok += correct
        op_name = ID_TO_OP.get(true_op, str(true_op))
        bucket = op_by_class.setdefault(op_name, {"correct": 0, "count": 0})
        bucket["correct"] += correct
        bucket["count"] += 1
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
        "op_accuracy_by_class": {
            name: {
                "accuracy": vals["correct"] / max(1, vals["count"]),
                "count": vals["count"],
            }
            for name, vals in sorted(op_by_class.items())
        },
        "pointer_examples": ptr_n,
        "pointer_hit_0p1": ptr_hit / max(1, ptr_n),
        "pointer_mean_l2": sum(ptr_l2) / max(1, len(ptr_l2)),
    }


def initialize_computer_text_from_main(base: ComputerUseV3, main_state: dict) -> int:
    """Transfer Main token semantics + compatible early Transformer weights."""
    token = main_state.get("token.weight")
    if token is None or tuple(token.shape) != tuple(base.task_emb.weight.shape):
        raise RuntimeError(
            f"ComputerUse/Main embedding mismatch: "
            f"main={None if token is None else tuple(token.shape)} "
            f"cu={tuple(base.task_emb.weight.shape)}"
        )
    with torch.no_grad():
        base.task_emb.weight.copy_(token.to(base.task_emb.weight.device, base.task_emb.weight.dtype))

        copied = 0
        for i, layer in enumerate(base.text_encoder.layers):
            prefix = f"blocks.{i}."
            required = [
                prefix + "attn.qkv.weight",
                prefix + "attn.proj.weight",
                prefix + "ln1.weight", prefix + "ln1.bias",
                prefix + "ln2.weight", prefix + "ln2.bias",
                prefix + "mlp.0.weight", prefix + "mlp.2.weight",
            ]
            if not all(k in main_state for k in required):
                break
            if tuple(main_state[prefix + "attn.qkv.weight"].shape) != tuple(layer.self_attn.in_proj_weight.shape):
                break
            layer.self_attn.in_proj_weight.copy_(
                main_state[prefix + "attn.qkv.weight"].to(
                    layer.self_attn.in_proj_weight.device,
                    layer.self_attn.in_proj_weight.dtype,
                )
            )
            layer.self_attn.out_proj.weight.copy_(
                main_state[prefix + "attn.proj.weight"].to(
                    layer.self_attn.out_proj.weight.device,
                    layer.self_attn.out_proj.weight.dtype,
                )
            )
            if layer.self_attn.in_proj_bias is not None:
                layer.self_attn.in_proj_bias.zero_()
            if layer.self_attn.out_proj.bias is not None:
                layer.self_attn.out_proj.bias.zero_()

            layer.norm1.weight.copy_(main_state[prefix + "ln1.weight"].to(layer.norm1.weight.device))
            layer.norm1.bias.copy_(main_state[prefix + "ln1.bias"].to(layer.norm1.bias.device))
            layer.norm2.weight.copy_(main_state[prefix + "ln2.weight"].to(layer.norm2.weight.device))
            layer.norm2.bias.copy_(main_state[prefix + "ln2.bias"].to(layer.norm2.bias.device))
            layer.linear1.weight.copy_(
                main_state[prefix + "mlp.0.weight"].to(layer.linear1.weight.device, layer.linear1.weight.dtype)
            )
            layer.linear2.weight.copy_(
                main_state[prefix + "mlp.2.weight"].to(layer.linear2.weight.device, layer.linear2.weight.dtype)
            )
            if layer.linear1.bias is not None:
                layer.linear1.bias.zero_()
            if layer.linear2.bias is not None:
                layer.linear2.bias.zero_()
            copied += 1
    return copied


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
        task_len=int(os.environ.get("FLM_V07_CU_TASK_LEN", "512")),
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
    main_embedding_inherited = False
    main_text_layers_inherited = 0
    if os.environ.get("FLM_V07_CU_INIT_FROM_MAIN", "1").strip().lower() not in {"0", "false", "no", "off"}:
        main_path = out_root / "main" / "checkpoint.pt"
        if not main_path.is_file():
            raise RuntimeError(
                "ComputerUse is configured to inherit Main text weights, but Main checkpoint.pt is missing. "
                "Train Main first or set FLM_V07_CU_INIT_FROM_MAIN=0 explicitly."
            )
        main_ck = torch.load(main_path, map_location="cpu", weights_only=False)
        main_text_layers_inherited = initialize_computer_text_from_main(
            base, main_ck.get("model") or {}
        )
        main_embedding_inherited = True
        if main_text_layers_inherited < cfg.text_layers:
            raise RuntimeError(
                f"ComputerUse inherited only {main_text_layers_inherited}/{cfg.text_layers} Main text layers"
            )
        print(
            f"V07_CU_TEXT_FROM_MAIN={main_path} layers={main_text_layers_inherited}",
            flush=True,
        )
    wrapped = base
    if runtime.kind == "gpu" and torch.cuda.device_count() > 1 and batch >= 2:
        wrapped = torch.nn.DataParallel(base)
    opt = torch.optim.AdamW(base.parameters(), lr=8e-5, betas=(0.9, 0.95), weight_decay=0.05)
    start = time.time()
    last = float("nan")
    action_counts = {op: len(rows) for op, rows in sampler.by_op.items()}
    stage = out_root / "computer_use"
    final_path = stage / "checkpoint.pt"
    resume_path = stage / "checkpoint_resume.pt"
    start_step = 0

    if resume_enabled() and final_path.is_file():
        done = torch.load(final_path, map_location=runtime.device, weights_only=False)
        done_result = done.get("result") or {}
        if int(done_result.get("steps", 0)) >= steps and done.get("config") == cfg.__dict__:
            base.load_state_dict(done["model"])
            print(f"V07_CU_ALREADY_COMPLETE steps={done_result.get('steps')}", flush=True)
            return done_result

    if resume_enabled() and resume_path.is_file():
        ck = torch.load(resume_path, map_location=runtime.device, weights_only=False)
        required = {"model", "optimizer", "sampler", "next_step"}
        if required.issubset(ck):
            if ck.get("config") != cfg.__dict__ or ck.get("domains") != domains:
                raise RuntimeError("ComputerUse resume config/domain mismatch")
            base.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            optimizer_to_device(opt, runtime.device)
            sampler.load_state_dict(ck["sampler"])
            start_step = int(ck["next_step"])
            last = float(ck.get("train_loss", float("nan")))
            print(f"V07_CU_RESUME step={start_step}/{steps}", flush=True)

    for step in range(start_step, steps):
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
        if (step + 1) % 2000 == 0:
            atomic_torch_save(
                {
                    "model": base.state_dict(),
                    "optimizer": opt.state_dict(),
                    "sampler": sampler.state_dict(),
                    "config": cfg.__dict__,
                    "domains": domains,
                    "step": step + 1,
                    "next_step": step + 1,
                    "train_loss": last,
                },
                resume_path,
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
        "task_embedding_initialized_from_main": main_embedding_inherited,
        "text_layers_initialized_from_main": main_text_layers_inherited,
        **ev,
        "elapsed_s": time.time() - start,
        "config": cfg.__dict__,
    }
    out = stage
    atomic_torch_save(
        {"model": base.state_dict(), "config": cfg.__dict__, "domains": domains, "result": result},
        final_path,
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

    text_version = os.environ.get("FLM_V07_TEXT_VERSION", "v0.7")
    computer_version = os.environ.get("FLM_V07_COMPUTER_VERSION", text_version)
    v07 = resolve_pipeline_root(text_version)
    computer_root = v07 if computer_version == text_version else resolve_pipeline_root(computer_version)
    print(f"v07_text_version={text_version} computer_version={computer_version}", flush=True)
    input_quality = validate_pipeline_inputs(v07, computer_root)
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
    main_steps = int(os.environ.get("FLM_V07_MAIN_STEPS", "0"))
    main_sft_steps = int(os.environ.get("FLM_V07_MAIN_SFT_STEPS", "0"))
    coder_steps = int(os.environ.get("FLM_V07_CODER_STEPS", "0"))
    coder_sft_steps = int(os.environ.get("FLM_V07_CODER_SFT_STEPS", "0"))
    cu_steps = int(os.environ.get("FLM_V07_CU_STEPS", "8000"))

    summary = {
        "pipeline_version": "v0.7",
        "accelerator": runtime.kind,
        "gpu_names": runtime.gpu_names,
        "tokenizer_vocab_size": vocab_size,
        "data_validation": input_quality,
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
        if args.only == "all":
            main_model = main_model.to("cpu")
            if runtime.kind == "gpu":
                torch.cuda.empty_cache()

    if args.only in {"all", "coder"}:
        coder_data = mmap_tokens(v07 / "coder_train.u16")
        coder_sft_data = mmap_tokens(v07 / "coder_sft_tokens.u16")
        coder_sft_mask = mmap_mask(v07 / "coder_sft_mask.u8")
        if len(coder_sft_data) != len(coder_sft_mask):
            raise RuntimeError("coder SFT token/mask mismatch")
        main_seed = None
        if os.environ.get("FLM_V07_CODER_INIT_FROM_MAIN", "1").strip().lower() not in {"0", "false", "no", "off"}:
            candidate = out / "main" / "checkpoint.pt"
            if candidate.is_file():
                main_seed = candidate
            else:
                raise RuntimeError(
                    "Coder is configured to inherit Main, but Main checkpoint.pt is missing. "
                    "Train Main first or set FLM_V07_CODER_INIT_FROM_MAIN=0 explicitly."
                )
        coder_model, coder_cfg, pre = train_text_pretrain(
            "coder", coder_data, out, runtime, vocab_size, coder_steps, batch, accum,
            init_checkpoint=main_seed,
        )
        coder_model, sft = train_text_sft(
            "coder", coder_model, coder_cfg, coder_sft_data, coder_sft_mask,
            out, runtime, coder_sft_steps, batch, accum,
        )
        summary["models"]["coder"] = {
            "pretrain": pre,
            "sft": sft,
            "initialized_from_main": bool(main_seed),
        }
        checkpoint_result(out, summary)

    if args.only == "all" and main_model is not None and coder_model is not None:
        main_model = main_model.to(runtime.device)
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
        cu = train_computer_use(computer_root, out, runtime, tok, vocab_size, cu_steps)
        summary["models"]["computer_use"] = cu
        checkpoint_result(out, summary)

    for name in ("tokenizer.json", "tokenizer_meta.json", "sources.json"):
        shutil.copy2(v07 / name, out / ("text_sources.json" if name == "sources.json" else name))
    if computer_root != v07 and (computer_root / "sources.json").is_file():
        shutil.copy2(computer_root / "sources.json", out / "computer_sources.json")
    quality_errors = []
    if args.only == "all":
        quality_errors = validate_suite(summary)
        summary["quality_gate"] = {
            "passed": not quality_errors,
            "errors": quality_errors,
        }
    else:
        summary["quality_gate"] = {
            "passed": None,
            "errors": [],
            "reason": f"partial training mode: {args.only}",
        }

    (out / "suite_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if quality_errors:
        (out / "quality_gate_errors.json").write_text(
            json.dumps(quality_errors, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("V07_QUALITY_GATE_FAILED", flush=True)
        for error in quality_errors:
            print(" - " + error, flush=True)
        raise RuntimeError(
            f"v0.7 quality gate failed with {len(quality_errors)} error(s)"
        )

    if args.only == "all":
        print("V07_QUALITY_GATE_PASSED", flush=True)
    print(f"V07_SUITE_COMPLETE output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
