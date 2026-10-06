#!/usr/bin/env python3
"""Synthetic DiT benchmark for LingBot-World-V2 1.3B causal-fast."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lingbot_accel.runtime import detect_accelerator, now_sync


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, default=Path("."), help="LingBot upstream clone")
    p.add_argument("--ckpt", type=Path, required=True, help="1.3B causal-fast checkpoint dir")
    p.add_argument("--backend", choices=["auto", "cuda", "xla", "cpu"], default="auto")
    p.add_argument("--max-area", type=int, default=256 * 448)
    p.add_argument("--aspect", type=float, default=16 / 9, help="width / height")
    p.add_argument("--chunk-size", type=int, default=1, help="latent frames per causal chunk")
    p.add_argument("--local-attn", type=int, default=6)
    p.add_argument("--sink", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def resolve_dit_dir(ckpt: Path) -> Path:
    for candidate in (ckpt / "transformers", ckpt):
        if (candidate / "config.json").exists() or list(candidate.glob("*.safetensors")):
            return candidate
    return ckpt


def load_state(dit_dir: Path):
    for idx_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        idx = dit_dir / idx_name
        if idx.exists():
            spec = json.loads(idx.read_text())
            out = {}
            for shard in sorted(set(spec["weight_map"].values())):
                out.update(load_file(str(dit_dir / shard)))
            return out
    for name in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
        path = dit_dir / name
        if path.exists():
            return load_file(str(path))
    raise FileNotFoundError(f"No safetensors found in {dit_dir}")


def main():
    a = parse_args()
    sys.path.insert(0, str(a.upstream.resolve()))

    from wan.configs import WAN_CONFIGS
    from wan.modules.model_fast import WanModelFast

    cfg = WAN_CONFIGS["i2v-1.3B"]
    acc = detect_accelerator(a.backend)
    dit_dir = resolve_dit_dir(a.ckpt)

    kwargs = dict(
        model_type="i2v",
        patch_size=tuple(cfg.patch_size),
        text_len=cfg.text_len,
        in_dim=getattr(cfg, "in_dim", 36),
        dim=cfg.dim,
        ffn_dim=cfg.ffn_dim,
        freq_dim=cfg.freq_dim,
        text_dim=getattr(cfg, "text_dim", 4096),
        out_dim=getattr(cfg, "out_dim", 16),
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        qk_norm=cfg.qk_norm,
        cross_attn_norm=cfg.cross_attn_norm,
        eps=cfg.eps,
        local_attn_size=a.local_attn,
        sink_size=a.sink,
    )

    print(f"[bench] backend={acc.kind} device={acc.device} ckpt={dit_dir}")
    model = WanModelFast(**kwargs)
    state = load_state(dit_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[bench] missing={len(missing)} unexpected={len(unexpected)}")
    del state

    model.eval().requires_grad_(False)
    model.to(dtype=torch.bfloat16)
    model.to(acc.device)

    h_over_w = 1.0 / a.aspect
    lat_h = round(math.sqrt(a.max_area * h_over_w) // cfg.vae_stride[1] // cfg.patch_size[1] * cfg.patch_size[1])
    lat_w = round(math.sqrt(a.max_area / h_over_w) // cfg.vae_stride[2] // cfg.patch_size[2] * cfg.patch_size[2])
    frame_seqlen = (lat_h * lat_w) // (cfg.patch_size[1] * cfg.patch_size[2])
    seq_len = a.chunk_size * frame_seqlen
    head_dim = cfg.dim // cfg.num_heads
    kv_size = frame_seqlen * a.local_attn if a.local_attn > -1 else frame_seqlen * a.chunk_size

    def z(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device=acc.device)

    kv_cache = [{
        "k": z((1, kv_size, cfg.num_heads, head_dim)),
        "v": z((1, kv_size, cfg.num_heads, head_dim)),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=acc.device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=acc.device),
    } for _ in range(cfg.num_layers)]

    cross_cache = [{
        "k": z((1, cfg.text_len, cfg.num_heads, head_dim)),
        "v": z((1, cfg.text_len, cfg.num_heads, head_dim)),
        "is_init": torch.tensor(0, dtype=torch.int32, device=acc.device),
    } for _ in range(cfg.num_layers)]

    latent = z((16, a.chunk_size, lat_h, lat_w), torch.float32)
    cond = z((20, a.chunk_size, lat_h, lat_w))
    cam = z((1, 6 * cfg.vae_stride[1] * cfg.vae_stride[2], a.chunk_size, lat_h, lat_w), cfg.param_dtype)
    context = z((cfg.text_len, 4096), cfg.param_dtype)
    timestep = torch.tensor([500.0], dtype=torch.float32, device=acc.device)

    kwargs_fwd = dict(
        x=[latent],
        t=timestep,
        context=[context],
        seq_len=seq_len,
        y=[cond],
        dit_cond_dict={"c2ws_plucker_emb": (cam,)},
        kv_cache=kv_cache,
        crossattn_cache=cross_cache,
        current_start=0,
        max_attention_size=kv_size,
        frame_seqlen=frame_seqlen,
    )

    times = []
    first = True
    total = a.warmup + a.iters
    with torch.no_grad(), acc.autocast(cfg.param_dtype):
        for i in range(total):
            t0 = now_sync(acc)
            _ = model(cross_attn_first_call=first, **kwargs_fwd)
            first = False
            t1 = now_sync(acc)
            dt = t1 - t0
            print(f"[bench] {'warmup' if i < a.warmup else 'measure'} {i+1}/{total}: {dt:.4f}s/forward")
            if i >= a.warmup:
                times.append(dt)

    mean_fwd = statistics.mean(times)
    result = {
        "backend": acc.kind,
        "device": str(acc.device),
        "max_area": a.max_area,
        "latent_hw": [lat_h, lat_w],
        "chunk_size_latent": a.chunk_size,
        "local_attn": a.local_attn,
        "sink": a.sink,
        "mean_forward_s": mean_fwd,
        "p50_forward_s": statistics.median(times),
        "estimated_core_generation_fps_excluding_vae": (a.chunk_size * cfg.vae_stride[0]) / (mean_fwd * 5.0),
        "forwards_per_chunk": 5,
    }
    print(json.dumps(result, indent=2))
    if a.json:
        a.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
