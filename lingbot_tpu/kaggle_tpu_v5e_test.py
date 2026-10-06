#!/usr/bin/env python3
"""End-to-end LingBot TPU smoke/latency runner for Kaggle TPU v5e-8.

Profiles:
- smoke: 480p-class sanity/quality path.
- interactive: 5-frame, 2-latent chunk, ~256x448 area for low latency.

The kernel bundle already contains the TPU patch scripts, so only upstream
LingBot and model weights are fetched at runtime.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
HERE = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'lingbot-world-v2'
CKPT = ROOT / 'lingbot-world-v2-1.3b-causal-fast'
ASSETS = ROOT / 'lingbot-world-v2-assets'


def run(cmd, cwd=None, env=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def ensure_checkout():
    if not UPSTREAM.exists():
        run([
            'git', 'clone', '--depth', '1',
            'https://github.com/Robbyant/lingbot-world-v2.git',
            UPSTREAM,
        ])


def ensure_python_deps():
    # Do not replace Kaggle's torch/torch_xla pair. Install only userspace deps.
    run([
        sys.executable, '-m', 'pip', 'install', '-q',
        '--disable-pip-version-check',
        'huggingface_hub[hf_xet]',
        'safetensors',
        'easydict',
        'ftfy',
        'imageio[ffmpeg]',
        'opencv-python-headless',
        'diffusers>=0.31.0',
        'transformers>=4.49.0,<=4.51.3',
        'accelerate>=1.1.1',
    ])
    try:
        import torch  # noqa: F401
        import torch_xla  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            'Kaggle TPU image must provide a matching torch + torch_xla pair; '
            'refusing to pip-install a potentially incompatible torch_xla build.'
        ) from exc


def download_models():
    from huggingface_hub import snapshot_download

    CKPT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id='robbyant/lingbot-world-v2-1.3b-causal-fast',
        local_dir=str(CKPT / 'transformers'),
    )
    # Shared T5/VAE/tokenizer only; do not download the 14B DiT.
    snapshot_download(
        repo_id='robbyant/lingbot-world-v2-14b-causal-fast',
        local_dir=str(ASSETS),
        allow_patterns=[
            'models_t5_umt5-xxl-enc-bf16.pth',
            'Wan2.1_VAE.pth',
            'google/**',
        ],
    )


def verify_tpu():
    os.environ.setdefault('PJRT_DEVICE', 'TPU')
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
    except Exception as exc:
        raise RuntimeError(
            'No real Kaggle TPU runtime detected. Open this notebook in the Kaggle editor, '
            'select Settings -> Accelerator -> TPU v5e-8, then Save & Run All.'
        ) from exc

    dev = xm.xla_device()
    print('torch:', torch.__version__)
    print('torch_xla:', getattr(torch_xla, '__version__', 'unknown'))
    print('TPU device:', dev)
    print('device_type:', xr.device_type())
    print('global_runtime_device_count:', xr.global_runtime_device_count())
    if str(xr.device_type()).upper() != 'TPU':
        raise RuntimeError('Kaggle runtime is not TPU. Select TPU v5e-8.')
    if xr.global_runtime_device_count() < 8:
        print('WARNING: fewer than 8 global TPU devices visible.')


def profile_values(args):
    if args.profile == 'interactive':
        return {
            'frames': args.frames or 5,
            'chunk_size': args.chunk_size or 2,
            'max_area': args.max_area or (256 * 448),
            'passes': args.passes or 3,
        }
    return {
        'frames': args.frames or 17,
        'chunk_size': args.chunk_size or 4,
        'max_area': args.max_area or 0,
        'passes': args.passes or 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', choices=['smoke', 'interactive'], default='smoke')
    ap.add_argument('--size', default='480*832',
                    choices=['480*832', '832*480', '720*1280', '1280*720'])
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=0)
    ap.add_argument('--max_area', type=int, default=0)
    ap.add_argument('--passes', type=int, default=0)
    ap.add_argument('--skip_download', action='store_true')
    args = ap.parse_args()
    vals = profile_values(args)

    # Verify the accelerator before downloads or pip work. The Kaggle CLI
    # push path can silently start a CPU image even when TPU is requested.
    verify_tpu()
    ensure_python_deps()
    ensure_checkout()
    if not args.skip_download:
        download_models()

    patch = HERE / 'apply_tpu_patch.py'
    generator = HERE / 'generate_tpu.py'
    run([sys.executable, patch, UPSTREAM])
    shutil.copy2(generator, UPSTREAM / 'generate_tpu.py')

    output = ROOT / (
        f'lingbot_tpu_{args.profile}_{args.size.replace("*","x")}_'
        f'{vals["frames"]}f.mp4'
    )
    cmd = [
        sys.executable, 'generate_tpu.py',
        '--ckpt_dir', str(CKPT),
        '--assets_dir', str(ASSETS),
        '--size', args.size,
        '--frame_num', str(vals['frames']),
        '--chunk_size', str(vals['chunk_size']),
        '--passes', str(vals['passes']),
        '--output', str(output),
    ]
    if vals['max_area'] > 0:
        cmd.extend(['--max_area', str(vals['max_area'])])

    print('PROFILE:', args.profile)
    print('BENCHMARK_CONFIG:', vals)
    run(cmd, cwd=UPSTREAM)


if __name__ == '__main__':
    main()
