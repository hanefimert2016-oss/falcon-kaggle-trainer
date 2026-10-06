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
os.environ.setdefault('XLA_AUTO_SPMD_MESH', '2,4')
os.environ.setdefault('XLA_AUTO_USE_GROUP_SHARDING', '1')

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from PIL import Image

# Auto-SPMD lets XLA choose sharding for a first functional port. Manual
# activation/weight sharding is the next optimization stage.
xr.use_spmd(auto=True)

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

    pipe, cfg = build_pipeline(args, device)

    # First pass compiles the graph. Do not call it a steady-state FPS result.
    compile_elapsed, frames, compile_fps, _ = run_once(pipe, cfg, args, device, save=False)
    print(f'COMPILE_RUN seconds={compile_elapsed:.3f} frames={frames} fps={compile_fps:.3f}')
    if args.warmup_only:
        return

    # Same shape/prompt path: this is the useful steady-state figure.
    elapsed, frames, fps, video = run_once(pipe, cfg, args, device, save=True)
    print(f'STEADY_STATE seconds={elapsed:.3f} frames={frames} fps={fps:.3f} size={args.size}')
    print(f'OUTPUT={args.output}')
    del video


if __name__ == '__main__':
    main()
