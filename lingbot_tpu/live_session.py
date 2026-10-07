#!/usr/bin/env python3
"""Stateful LingBot interactive session.

This keeps the LingBot pipeline resident on the accelerator and turns each
WASD update into a 5-frame I2V request (1 conditioning frame + 4 newly
generated frames). The last generated frame becomes the next conditioning
frame, so the world can continue interactively without rebuilding the model.

Phase 1 intentionally reuses upstream generate() for correctness. XLA graphs,
weights and T5 prompt embeddings stay warm. Phase 2 can lift the causal
self-KV cache across step() calls after the TPU benchmark validates the base
port.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from wasd_camera import WasdCamera, write_action_dir


@dataclass
class StepStats:
    seconds: float
    generated_frames: int
    fps: float
    max_area: int
    keys: tuple[str, ...]


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert LingBot [-1,1] CHW frame tensor to RGB PIL without surprises."""
    x = frame.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"expected CHW RGB frame, got {tuple(x.shape)}")
    return Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")


class LingBotLiveSession:
    """Persistent model + camera state for low-latency interactive chunks."""

    def __init__(
        self,
        pipe,
        cfg,
        *,
        initial_image: Image.Image,
        prompt: str,
        intrinsics_source: str | Path,
        work_dir: str | Path = "/kaggle/working/lingbot-live",
        max_area: int = 256 * 448,
        chunk_size: int = 2,
        seed: int = 42,
        move_per_frame: float = 0.08,
        yaw_per_frame_deg: float = 2.0,
    ):
        self.pipe = pipe
        self.cfg = cfg
        self.image = initial_image.convert("RGB")
        self.prompt = str(prompt)
        self.intrinsics_source = Path(intrinsics_source)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.action_dir = self.work_dir / "action"
        self.max_area = int(max_area)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.camera = WasdCamera(
            move_per_frame=move_per_frame,
            yaw_per_frame_deg=yaw_per_frame_deg,
        )
        self.step_index = 0

        if self.chunk_size != 2:
            # frame_num=5 gives two latent frames for Wan VAE stride=4.
            raise ValueError("interactive 5-frame mode currently requires chunk_size=2")

    def step(
        self,
        keys=(),
        *,
        mouse_dx: float = 0.0,
        mouse_dy: float = 0.0,
        max_area: int | None = None,
    ):
        keys_tuple = tuple(sorted(str(k).lower() for k in keys))
        area = int(max_area or self.max_area)

        poses = self.camera.trajectory(
            keys_tuple,
            frame_num=5,
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
        )
        write_action_dir(
            self.action_dir,
            poses,
            intrinsics_source=self.intrinsics_source,
        )

        # Same seed keeps benchmark/control comparisons reproducible. Offset by
        # step so repeated chunks do not receive identical initial noise.
        seed = self.seed + self.step_index

        # Synchronize only around the measured generation boundary. On XLA,
        # generate_tpu's caller can additionally xm.mark_step/wait_device_ops.
        t0 = time.perf_counter()
        video = self.pipe.generate(
            self.prompt,
            self.image,
            action_path=str(self.action_dir),
            chunk_size=self.chunk_size,
            max_area=area,
            frame_num=5,
            shift=self.cfg.sample_shift,
            seed=seed,
            offload_model=False,
            max_attention_size=None,
        )

        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
            xm.wait_device_ops()
        except Exception:
            # CUDA/CPU smoke compatibility.
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        if video is None:
            raise RuntimeError("LingBot returned no video on rank 0")

        # Upstream output is [C,F,H,W]. Keep the final frame as the next I2V
        # conditioning image; return only the four newly generated frames to UI.
        if video.ndim != 4:
            raise RuntimeError(f"unexpected LingBot video shape {tuple(video.shape)}")
        frames = int(video.shape[1])
        generated = max(frames - 1, 1)
        self.image = tensor_frame_to_pil(video[:, -1])
        self.step_index += 1

        stats = StepStats(
            seconds=elapsed,
            generated_frames=generated,
            fps=generated / elapsed,
            max_area=area,
            keys=keys_tuple,
        )
        return video[:, 1:], stats

    def warmup(self, keys=("w",)):
        """Compile/warm a same-shape interactive chunk, then keep the pipe hot."""
        return self.step(keys)


__all__ = ["LingBotLiveSession", "StepStats", "tensor_frame_to_pil"]
