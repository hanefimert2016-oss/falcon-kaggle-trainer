# LingBot-World-V2 1.3B — 8GB CUDA + TPU experiment v0.1

This is a small patch/benchmark layer on top of the official `Robbyant/lingbot-world-v2` repository.
It does **not** redistribute model weights.

## Goal

- RTX 4060 Laptop 8GB: run the 1.3B causal-fast model with T5 on CPU, short KV window and a reduced `max_area`, then display at 1280×720 / 16 FPS.
- Kaggle TPU v5e: first prove that the 1.3B DiT core loads/compiles/runs under PyTorch/XLA, then measure per-forward latency before attempting multi-core sharding.

## Why lower `max_area` instead of only INT4?

On this model the temporal KV cache is a major memory consumer. Shrinking the causal window and latent spatial area attacks the actual memory hot spot. The 1.3B DiT itself is already much smaller than the 14B model.

## RTX 4060 path

From an official upstream clone:

```bash
python /path/to/lingbot-world-v2-lowvram-tpu/scripts/patch_lowvram.py .

CKPT=./lingbot-world-v2-1.3b-causal-fast \
ASSETS=./lingbot-world-v2-14b-causal-fast \
/path/to/lingbot-world-v2-lowvram-tpu/scripts/run_4060_8gb.sh
```

Start with `AREA=114688` (`256*448` max area), `LOCAL_ATTN=6`, `SINK=2`, `CHUNK=1`.
If peak VRAM is comfortably below 8GB, try `AREA=174080` (`320*544`).

The model output can then be converted to a 720p/16 FPS display stream without extra neural-model VRAM:

```bash
python scripts/upscale_720p16.py input.mp4 output_720p16.mp4
```

## Synthetic CUDA/TPU DiT benchmark

```bash
python scripts/bench_dit.py \
  --upstream /path/to/lingbot-world-v2 \
  --ckpt /path/to/lingbot-world-v2-1.3b-causal-fast \
  --backend cuda \
  --max-area 114688 --chunk-size 1 --local-attn 6 --sink 2
```

The reported `estimated_core_generation_fps_excluding_vae` assumes the official causal-fast budget of four denoise forwards plus one clean-latent KV refresh per chunk. It is **not** end-to-end FPS; VAE decode, I/O and display are intentionally excluded.

## Kaggle TPU

Enable a TPU accelerator in Kaggle and run:

```bash
python kaggle/tpu_v5e_smoke.py
```

The TPU smoke gate benchmarks only the DiT core first. Full T5/VAE TPU support and 8-core GSPMD partitioning are next-stage work after the operator-compatibility gate passes.

## Current limits

- v0.1 does not claim native 1280×720 generation at 16 AI FPS on an 8GB RTX 4060.
- v0.1 display 720p/16 uses resize + cheap temporal interpolation; it is not neural RIFE yet.
- TPU benchmark is single-device/core-oriented first. Multi-core sharding is deliberately postponed until the model executes correctly on XLA.
