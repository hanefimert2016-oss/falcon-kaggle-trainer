#!/usr/bin/env python3
"""LingBot-World V2 1.3B causal-fast inference on PyTorch/XLA TPU.

Run this *inside the patched upstream repository*.
Uses one logical SPMD XLA device; on Kaggle v5e-8 XLA auto-sharding can
partition the graph across all eight chips.
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

# Enable SPMD before any XLA tensors are created. Auto-sharding handles the
# DiT/VAE graph, while a named 1-D 8-core mesh is also exposed to the Pallas
# attention wrapper for explicit head parallelism.
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
    p.add_argument('--frame_num', type=int, default=17)
    p.add_argument('--chunk_size', type=int, default=4)
    p.add_argument('--local_attn_size', type=int, default=18)
    p.add_argument('--sink_size', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default='A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky.')
    p.add_argument('--output', default='output/lingbot_tpu.mp4')
    p.add_argument('--warmup_only', action='store_true')
    p.add_argument('--steady_runs', type=int, default=1,
                   help='Number of same-shape steady-state runs after compilation.')
    p.add_argument('--no_save', action='store_true',
                   help='Benchmark without writing the final MP4.')
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


def run_once(pipe, cfg, args, device, save=False):
    img = Image.open(args.image).convert('RGB')
    xm.mark_step()
    xm.wait_device_ops()
    t0 = time.perf_counter()
    video = pipe.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=args.chunk_size,
        max_area=MAX_AREA_CONFIGS[args.size],
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
    fps = frames / elapsed
    if save and video is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_video(video[None], args.output, fps=cfg.sample_fps, nrow=1,
                   normalize=True, value_range=(-1, 1))
    return elapsed, frames, fps, video


def main():
    args = parse_args()
    if (args.frame_num - 1) % 4 != 0:
        raise SystemExit('--frame_num must be 4n+1 (e.g. 5, 17, 81)')

    device = xm.xla_device()
    print('XLA device:', device)
    print('XLA device kind:', xr.device_type())
    print('Global runtime devices:', xr.global_runtime_device_count())
    print('SPMD:', xr.is_spmd())
    print('Auto mesh:', os.environ.get('XLA_AUTO_SPMD_MESH'))
    print('Pallas global mesh:', _GLOBAL_MESH)

    pipe, cfg = build_pipeline(args, device)

    # First pass compiles the graph. Do not call it a steady-state FPS result.
    compile_elapsed, frames, compile_fps, _ = run_once(pipe, cfg, args, device, save=False)
    print(f'COMPILE_RUN seconds={compile_elapsed:.3f} frames={frames} fps={compile_fps:.3f}')
    if args.warmup_only:
        return

    # Same shape/prompt path: report all runs plus median steady-state FPS.
    steady = []
    last_video = None
    runs = max(1, args.steady_runs)
    for idx in range(runs):
        should_save = (idx == runs - 1) and (not args.no_save)
        elapsed, frames, fps, video = run_once(
            pipe, cfg, args, device, save=should_save)
        steady.append((elapsed, frames, fps))
        last_video = video
        print(f'STEADY_RUN index={idx} seconds={elapsed:.3f} frames={frames} fps={fps:.3f} size={args.size}')

    fps_values = sorted(v[2] for v in steady)
    elapsed_values = sorted(v[0] for v in steady)
    mid = len(fps_values) // 2
    if len(fps_values) % 2:
        median_fps = fps_values[mid]
        median_elapsed = elapsed_values[mid]
    else:
        median_fps = 0.5 * (fps_values[mid - 1] + fps_values[mid])
        median_elapsed = 0.5 * (elapsed_values[mid - 1] + elapsed_values[mid])

    print(f'STEADY_STATE_MEDIAN seconds={median_elapsed:.3f} frames={steady[-1][1]} fps={median_fps:.3f} size={args.size} runs={runs}')
    if not args.no_save:
        print(f'OUTPUT={args.output}')

    try:
        import torch_xla.debug.metrics as met
        print('XLA_METRICS_BEGIN')
        print(met.metrics_report())
        print('XLA_METRICS_END')
    except Exception as exc:
        print(f'XLA_METRICS_UNAVAILABLE={exc}')

    del last_video


if __name__ == '__main__':
    main()
