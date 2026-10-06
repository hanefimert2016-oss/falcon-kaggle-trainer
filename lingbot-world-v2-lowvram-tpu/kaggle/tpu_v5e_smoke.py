#!/usr/bin/env python3
"""Kaggle TPU v5e smoke benchmark for LingBot-World-V2 1.3B DiT."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
UP = ROOT / "lingbot-world-v2"
CKPT = ROOT / "lingbot-world-v2-1.3b-causal-fast"
PATCH_DEFAULT = ROOT / "lingbot-lowvram-tpu"


def run(*cmd, cwd=None):
    print("+", *map(str, cmd), flush=True)
    subprocess.check_call([str(x) for x in cmd], cwd=cwd)


def main():
    try:
        import torch_xla.core.xla_model as xm
        print("XLA device:", xm.xla_device())
    except Exception as e:
        raise SystemExit(f"TPU/XLA unavailable: {e}")

    if not UP.exists():
        run("git", "clone", "--depth", "1", "https://github.com/Robbyant/lingbot-world-v2.git", UP)

    run(sys.executable, "-m", "pip", "install", "-q",
        "safetensors", "huggingface_hub[cli]", "opencv-python-headless",
        "diffusers>=0.31", "transformers>=4.49,<=4.51.3", "tokenizers>=0.20.3",
        "accelerate>=1.1.1", "easydict", "einops", "ftfy", "imageio", "scipy", "numpy<2")

    if not CKPT.exists():
        run("huggingface-cli", "download", "robbyant/lingbot-world-v2-1.3b-causal-fast", "--local-dir", CKPT)

    patch_dir = PATCH_DEFAULT
    if not patch_dir.exists():
        repo = os.environ.get("LOWVRAM_REPO", "https://github.com/hanefimert2016-oss/falcon-kaggle-trainer.git")
        run("git", "clone", "--depth", "1", "--branch",
            os.environ.get("LOWVRAM_BRANCH", "lingbot-world-v2-tpu-v0.1"),
            repo, ROOT / "trainer")
        patch_dir = ROOT / "trainer" / "lingbot-world-v2-lowvram-tpu"

    run(sys.executable, patch_dir / "scripts" / "bench_dit.py",
        "--upstream", UP,
        "--ckpt", CKPT,
        "--backend", "xla",
        "--max-area", str(256 * 448),
        "--chunk-size", "1",
        "--local-attn", "6",
        "--sink", "2",
        "--warmup", "1",
        "--iters", "3",
        "--json", ROOT / "lingbot_tpu_bench.json")


if __name__ == "__main__":
    main()
