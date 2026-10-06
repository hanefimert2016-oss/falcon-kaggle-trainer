#!/usr/bin/env python3
"""End-to-end setup + smoke/benchmark runner for a Kaggle TPU v5e-8 notebook.

Expected notebook setting: Settings -> Accelerator -> TPU v5e-8, Internet ON.
This script is intentionally restart-safe: downloads and clones are cached.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
UPSTREAM = ROOT / 'lingbot-world-v2'
PORT = ROOT / 'lingbot-tpu-port'
CKPT = ROOT / 'lingbot-world-v2-1.3b-causal-fast'
ASSETS = ROOT / 'lingbot-world-v2-assets'


def run(cmd, cwd=None, env=None):
    print('+', ' '.join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def ensure_checkout():
    if not UPSTREAM.exists():
        run(['git', 'clone', '--depth', '1', 'https://github.com/Robbyant/lingbot-world-v2.git', UPSTREAM])
    if not PORT.exists():
        run(['git', 'clone', '--depth', '1', '--branch', 'lingbot-tpu-v5e-port',
             'https://github.com/hanefimert2016-oss/falcon-kaggle-trainer.git', PORT])


def ensure_python_deps():
    # Keep Kaggle's matching torch/torch-xla pair if already present.
    run([sys.executable, '-m', 'pip', 'install', '-q', '-U',
         'huggingface_hub[hf_xet]', 'safetensors', 'easydict', 'ftfy',
         'imageio[ffmpeg]', 'opencv-python-headless',
         'diffusers>=0.31.0', 'transformers>=4.49.0,<=4.51.3',
         'accelerate>=1.1.1'])
    try:
        import torch_xla  # noqa: F401
    except Exception:
        run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_xla[pallas]'])


def download_models():
    from huggingface_hub import snapshot_download
    CKPT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    # 1.3B repository is DiT-only; upstream expects it under /transformers.
    snapshot_download(
        repo_id='robbyant/lingbot-world-v2-1.3b-causal-fast',
        local_dir=str(CKPT / 'transformers'),
        local_dir_use_symlinks=False,
    )
    # Pull only shared T5/VAE/tokenizer assets, not the 14B DiT shards.
    snapshot_download(
        repo_id='robbyant/lingbot-world-v2-14b-causal-fast',
        local_dir=str(ASSETS),
        allow_patterns=['models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth', 'google/**'],
        local_dir_use_symlinks=False,
    )


def verify_tpu():
    os.environ.setdefault('PJRT_DEVICE', 'TPU')
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    print('TPU device:', dev)
    print('device_type:', xr.device_type())
    print('global_runtime_device_count:', xr.global_runtime_device_count())
    if str(xr.device_type()).upper() != 'TPU':
        raise RuntimeError('Kaggle runtime is not a TPU. Select TPU v5e-8 in notebook settings.')
    if xr.global_runtime_device_count() < 8:
        print('WARNING: fewer than 8 TPU devices visible; benchmark will not represent v5e-8.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default='480*832', choices=['480*832', '832*480', '720*1280', '1280*720'])
    ap.add_argument('--frames', type=int, default=17)
    ap.add_argument('--skip_download', action='store_true')
    args = ap.parse_args()

    ensure_python_deps()
    verify_tpu()
    ensure_checkout()
    if not args.skip_download:
        download_models()

    patch = PORT / 'lingbot_tpu' / 'apply_tpu_patch.py'
    generator = PORT / 'lingbot_tpu' / 'generate_tpu.py'
    run([sys.executable, patch, UPSTREAM])
    shutil.copy2(generator, UPSTREAM / 'generate_tpu.py')

    cmd = [
        sys.executable, 'generate_tpu.py',
        '--ckpt_dir', str(CKPT),
        '--assets_dir', str(ASSETS),
        '--size', args.size,
        '--frame_num', str(args.frames),
        '--output', str(ROOT / f'lingbot_tpu_{args.size.replace("*","x")}_{args.frames}f.mp4'),
    ]
    run(cmd, cwd=UPSTREAM)


if __name__ == '__main__':
    main()
