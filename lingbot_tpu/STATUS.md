# LingBot World V2 — Kaggle TPU Port Status

Updated: 2026-10-07

## Goal

Run `robbyant/lingbot-world-v2-1.3b-causal-fast` as a low-latency,
WASD-controlled world-model experience on Kaggle TPU, with a target of
16 generated FPS and the highest resolution that can sustain it.

## Code status

- PyTorch/XLA TPU portability patch: PASS
- Pallas FlashAttention path: PASS (static/preflight)
- 8-device SPMD mesh setup: PASS (static/preflight)
- CUDA-only autocast/cache/sync removal for TPU path: PASS
- T5 CPU path: PASS (static/preflight)
- VAE CPU-first load / XLA transfer path: PASS (static/preflight)
- WASD -> c2w `poses.npy` trajectory generation: PASS
- WASD preflight movement/intrinsics checks: PASS
- Model-resident `LingBotLiveSession`: PASS (static/preflight)
- Adaptive 16-FPS resolution ladder: PASS (static/preflight)

Latest preflight covered current upstream `Robbyant/lingbot-world-v2` plus
all `lingbot_tpu/*.py` files.

## Interactive benchmark profile

First target:

- 5 output frames per request: 1 conditioning + 4 newly generated
- causal latent chunk size: 2
- first max area: 114,688 pixels (~256x448)
- model remains resident
- T5 prompt cache remains warm
- last generated frame feeds the next interactive chunk

Adaptive ladder after the first real TPU allocation:

1. 114,688 (~256x448)
2. 184,320 (~320x576)
3. 230,400 (~360x640)
4. 399,360 (480p class)
5. 921,600 (720p class)

Each area gets compile/warmup + steady-state passes. The ladder stops at the
first area whose average generated FPS is below 16 and reports the highest
area that met the 16-FPS target.

## Kaggle allocation tests

### TPU v5e-8 batch

A self-contained Kaggle kernel was successfully pushed. It remained QUEUED
for the entire watcher window and never reserved a TPU worker.

### TPU v5e-8 direct interactive session

CreateKernelSession accepted the request and returned session 355946706.
The operation remained `done=False`, TPU `time_reserved=0`, and returned no
allocator error for the full polling window. It was cancelled cleanly.

### Legacy v3 alias -> current TPU allocator

`Tpu1VmV38` was accepted for interactive allocation, session 355948947.
It also remained unallocated (`time_reserved=0`) for the full probe and was
cancelled cleanly.

### TPU v6e-8 direct interactive session

`TpuV6E8` was accepted, session 355949622. It remained unallocated
(`time_reserved=0`) for the 60-second probe and was cancelled cleanly.

### TPU v6e-8 batch

Self-contained v6e kernel push succeeded. Kaggle reported
`KernelWorkerStatus.QUEUED` on all 60 polls over roughly ten minutes.
No TPU runtime started, so there is still no valid TPU FPS measurement.

## Quota

At the last quota probe:

- TPU total: 20:00:00
- TPU used: 00:04:56.042
- TPU reserved: 00:00:00
- quota refresh: 2026-10-10 00:00:00

Therefore quota is not the blocker. The current blocker is TPU worker
availability/allocation.

## Benchmark validity rules

Do not report an FPS result as a Kaggle TPU result unless runtime validation
shows:

- `torch_xla` imports successfully
- `xr.device_type() == "TPU"`
- at least 8 global XLA runtime devices are visible
- at least one steady-state run completes after initial XLA compilation

Only `STEADY_SUMMARY` / ladder steady-state output is used for FPS.

## Next execution when a worker is allocated

1. Verify actual TPU + 8 devices.
2. Download 1.3B DiT and shared T5/VAE assets.
3. Apply TPU patch.
4. Run WASD camera-controlled 5-frame warmup.
5. Measure steady generated FPS at ~256x448.
6. Run adaptive resolution ladder toward 720p/16 FPS.
7. If throughput is adequate, switch from repeated I2V feedback to the
   experimental persistent causal KV-cache streaming path.
8. Use the live session core for continuous WASD chunks.
