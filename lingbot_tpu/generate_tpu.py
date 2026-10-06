#!/usr/bin/env python3
"""LingBot-World V2 1.3B causal-fast inference on PyTorch/XLA TPU.

Run this *inside the patched upstream repository*.

Goals:
- one logical SPMD XLA device spanning Kaggle TPU v5e-8,
- BF16 execution,
- persistent XLA compilation cache,
- compile-run vs steady-state FPS separation,
- low-latency interactive benchmark mode via smaller max_area/chunk size.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Must be set before XLA runtime initialization.
os.environ.setdefault('PJRT_DEVICE', 'TPU')
os.environ.setdefault('XLA_USE_BF16', '1')
os.environ.setdefault('XLA_AUTO_SPMD_MESH', '8,1')
os.environ.setdefault('XLA_AUTO_USE_GROUP_SHARDING', '1')

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from PIL import Image

# Enable SPMD before creating XLA tensors. The 1-D mesh is used by the
# Pallas attention patch for 8-way head parallelism; auto-sharding remains
# enabled for the rest of the DiT/VAE graph.
xr.use_spmd(auto=True)
_NDEV = xr.global_runtime_device_count()
_GLOBAL_MESH = Mesh(np.arange(_NDEV), (_NDEV,), ('model',))
xs.set_global_mesh(_GLOBAL_MESH)

import wan
from wan.configs import MAX_AREA_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--assets_dir', required=True)
    p.add_argument('--image', default='examples/03/image.jpg')
    p.add_argument('--action_path', default='examples/03')
    p.add_argument('--size', default='480*832', choices=SUPPORTED_SIZES['i2v-1.3B'])
    p.add_argument(
        '--max_area',
        type=int,
        default=0,
        help='Override pixel area used by LingBot. Example: 114688 ~= 256*448. '
             '0 keeps the selected --size area.'
    )
    p.add_argument('--frame_num', type=int, default=17)
    p.add_argument('--chunk_size', type=int, default=4)
    p.add_argument('--local_attn_size', type=int, default=18)
    p.add_argument('--sink_size', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--passes', type=int, default=3,
                   help='Total identical-shape passes. Pass 1 compiles; later passes measure steady state.')
    p.add_argument('--xla_cache', default='/kaggle/working/xla_compile_cache')
    p.add_argument('--no_save', action='store_true')
    p.add_argument('--prompt', default='A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky.')
    p.add_argument('--output', default='output/lingbot_tpu.mp4')
    p.add_argument('--warmup_only', action='store_true')
    return p.parse_args()


def build_pipeline(args, device):
    cfg = WAN_CONFIGS['i2v-1.3B']
    pipe = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=False,
        convert_model_dtype=True,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode='causal_fast',
        assets_dir=args.assets_dir,
    )
    return pipe, cfg


def memory_snapshot(device, label):
    try:
        info = xm.get_memory_info(device)
        print(f'MEMORY {label}: {info}', flush=True)
    except Exception as exc:
        print(f'MEMORY {label}: unavailable ({exc})', flush=True)


def run_once(pipe, cfg, args, device, save=False):
    img = Image.open(args.image).convert('RGB')
    xm.mark_step()
    xm.wait_device_ops()

    max_area = args.max_area if args.max_area > 0 else MAX_AREA_CONFIGS[args.size]
    t0 = time.perf_counter()
    video = pipe.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=args.chunk_size,
        max_area=max_area,
        frame_num=args.frame_num,
        shift=cfg.sample_shift,
        seed=args.seed,
        offload_model=False,
        max_attention_size=None,
    )
    xm.mark_step()
    xm.wait_device_ops()
    elapsed = time.perf_counter() - t0

    frames = int(video.shape[1]) if video is not None and video.ndim >= 2 else args.frame_num
    # I2V returns the conditioning/initial frame as part of the clip. For
    # real-time world-model throughput, count only newly generated frames.
    generated_frames = max(frames - 1, 1)
    fps = generated_frames / elapsed
    if save and video is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_video(
            video[None],
            args.output,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    return elapsed, frames, fps, video


def main():
    args = parse_args()
    if (args.frame_num - 1) % 4 != 0:
        raise SystemExit('--frame_num must be 4n+1 (e.g. 5, 9, 13, 17, 81)')
    if args.chunk_size < 1:
        raise SystemExit('--chunk_size must be >= 1')
    if args.passes < 1:
        raise SystemExit('--passes must be >= 1')

    # Persistent compilation cache must be initialized before XLA computation.
    if args.xla_cache:
        Path(args.xla_cache).mkdir(parents=True, exist_ok=True)
        xr.initialize_cache(args.xla_cache)

    device = xm.xla_device()
    print('XLA device:', device)
    print('XLA device kind:', xr.device_type())
    print('Global runtime devices:', xr.global_runtime_device_count())
    print('SPMD:', xr.is_spmd())
    print('Auto mesh:', os.environ.get('XLA_AUTO_SPMD_MESH'))
    print('Pallas global mesh:', _GLOBAL_MESH)
    print('Requested frame_num:', args.frame_num)
    print('Chunk size:', args.chunk_size)
    print('Requested max_area:', args.max_area if args.max_area else MAX_AREA_CONFIGS[args.size])
    print('XLA cache:', args.xla_cache)
    memory_snapshot(device, 'before_pipeline')

    pipe, cfg = build_pipeline(args, device)
    xm.mark_step()
    xm.wait_device_ops()
    memory_snapshot(device, 'after_pipeline')

    steady_fps = []
    steady_seconds = []
    total_passes = 1 if args.warmup_only else args.passes

    for idx in range(total_passes):
        save = (idx == total_passes - 1) and (not args.no_save) and (not args.warmup_only)
        elapsed, frames, fps, video = run_once(pipe, cfg, args, device, save=save)
        if idx == 0:
            print(f'COMPILE_RUN seconds={elapsed:.3f} frames={frames} fps={fps:.3f}', flush=True)
        else:
            steady_fps.append(fps)
            steady_seconds.append(elapsed)
            print(
                f'STEADY_STATE_{idx} seconds={elapsed:.3f} frames={frames} '
                f'fps={fps:.3f} size={args.size} max_area={args.max_area or MAX_AREA_CONFIGS[args.size]} '
                f'chunk={args.chunk_size}',
                flush=True,
            )
        memory_snapshot(device, f'after_pass_{idx}')
        del video

    if steady_fps:
        avg_fps = sum(steady_fps) / len(steady_fps)
        avg_seconds = sum(steady_seconds) / len(steady_seconds)
        print(
            f'STEADY_SUMMARY passes={len(steady_fps)} avg_seconds={avg_seconds:.3f} '
            f'avg_fps={avg_fps:.3f} best_fps={max(steady_fps):.3f}',
            flush=True,
        )
    if not args.no_save and not args.warmup_only:
        print(f'OUTPUT={args.output}', flush=True)


if __name__ == '__main__':
    main()
