#!/usr/bin/env python3
"""WASD -> LingBot camera trajectory helper.

LingBot World V2's released 1.3B causal-fast inference path consumes
camera-to-world poses from poses.npy plus intrinsics.npy. This module turns
live keyboard/mouse state into short 4n+1 camera paths suitable for
low-latency chunked inference.

Coordinate convention follows the released camera utilities: c2w rotations
map camera rays [x right, y down, z forward] into world space.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


@dataclass
class CameraState:
    position: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0

    @classmethod
    def identity(cls) -> "CameraState":
        return cls(np.zeros(3, dtype=np.float32))

    def rotation(self) -> np.ndarray:
        # Camera local +Z is forward. Clamp pitch to avoid singular flips.
        p = float(np.clip(self.pitch, -math.radians(85), math.radians(85)))
        return (_ry(self.yaw) @ _rx(p)).astype(np.float32)

    def matrix(self) -> np.ndarray:
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = self.rotation()
        out[:3, 3] = self.position
        return out


class WasdCamera:
    """Stateful short-horizon camera controller for interactive generation."""

    def __init__(
        self,
        move_per_frame: float = 0.08,
        yaw_per_frame_deg: float = 2.0,
        pitch_per_frame_deg: float = 1.5,
        state: CameraState | None = None,
    ):
        self.move_per_frame = float(move_per_frame)
        self.yaw_per_frame = math.radians(float(yaw_per_frame_deg))
        self.pitch_per_frame = math.radians(float(pitch_per_frame_deg))
        self.state = state or CameraState.identity()

    @staticmethod
    def _pressed(keys, name: str) -> bool:
        if isinstance(keys, dict):
            return bool(keys.get(name, False) or keys.get(name.upper(), False))
        return name.lower() in {str(k).lower() for k in keys}

    def step(self, keys=(), mouse_dx: float = 0.0, mouse_dy: float = 0.0) -> np.ndarray:
        # Arrow keys or mouse rotate. A/D remain strafe controls.
        if self._pressed(keys, "left"):
            self.state.yaw -= self.yaw_per_frame
        if self._pressed(keys, "right"):
            self.state.yaw += self.yaw_per_frame
        if self._pressed(keys, "up"):
            self.state.pitch -= self.pitch_per_frame
        if self._pressed(keys, "down"):
            self.state.pitch += self.pitch_per_frame

        self.state.yaw += float(mouse_dx) * self.yaw_per_frame * 0.08
        self.state.pitch += float(mouse_dy) * self.pitch_per_frame * 0.08
        self.state.pitch = float(np.clip(
            self.state.pitch, -math.radians(85), math.radians(85)))

        R = self.state.rotation()
        right = R[:, 0]
        down = R[:, 1]
        forward = R[:, 2]

        delta = np.zeros(3, dtype=np.float32)
        if self._pressed(keys, "w"):
            delta += forward
        if self._pressed(keys, "s"):
            delta -= forward
        if self._pressed(keys, "d"):
            delta += right
        if self._pressed(keys, "a"):
            delta -= right
        # Space rises in world view; Ctrl/C falls.
        if self._pressed(keys, "space"):
            delta -= down
        if self._pressed(keys, "ctrl") or self._pressed(keys, "c"):
            delta += down

        norm = float(np.linalg.norm(delta))
        if norm > 1e-8:
            delta /= norm
            self.state.position = (
                self.state.position + delta * self.move_per_frame
            ).astype(np.float32)

        return self.state.matrix()

    def trajectory(
        self,
        keys=(),
        *,
        frame_num: int = 5,
        mouse_dx: float = 0.0,
        mouse_dy: float = 0.0,
    ) -> np.ndarray:
        if frame_num < 5 or (frame_num - 1) % 4:
            raise ValueError("LingBot frame_num must be 4n+1; interactive mode expects >=5")

        poses = [self.state.matrix()]
        n_new = frame_num - 1
        for i in range(n_new):
            # Spread one sampled mouse delta over all generated frames so a
            # chunk is a smooth trajectory rather than a single rotation jump.
            poses.append(self.step(
                keys,
                mouse_dx=mouse_dx / max(n_new, 1),
                mouse_dy=mouse_dy / max(n_new, 1),
            ))
        return np.stack(poses, axis=0).astype(np.float32)


def write_action_dir(
    out_dir: str | Path,
    poses: np.ndarray,
    *,
    intrinsics_source: str | Path | None = None,
    intrinsics: np.ndarray | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape [F,4,4], got {poses.shape}")
    np.save(out / "poses.npy", poses)

    if intrinsics_source is not None:
        K = np.asarray(np.load(intrinsics_source), dtype=np.float32)
    elif intrinsics is not None:
        K = np.asarray(intrinsics, dtype=np.float32)
    else:
        # Conservative 60-degree horizontal-FOV default for LingBot's
        # reference 832x480 coordinate system. Prefer copying the original
        # example intrinsics whenever available.
        w, h = 832.0, 480.0
        fx = 0.5 * w / math.tan(math.radians(60.0) / 2.0)
        fy = fx
        K = np.array([[fx, fy, w / 2.0, h / 2.0]], dtype=np.float32)

    if K.ndim == 1:
        K = K[None, :]
    if K.shape[-1] != 4:
        raise ValueError(f"intrinsics must end in 4 values [fx,fy,cx,cy], got {K.shape}")
    np.save(out / "intrinsics.npy", K)
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/lingbot-live-action")
    p.add_argument("--keys", default="w", help="comma-separated keys, e.g. w,d,right")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--intrinsics-source")
    args = p.parse_args()

    keys = [x.strip() for x in args.keys.split(",") if x.strip()]
    ctl = WasdCamera()
    poses = ctl.trajectory(keys, frame_num=args.frames)
    out = write_action_dir(
        args.out, poses,
        intrinsics_source=args.intrinsics_source,
    )
    print(f"ACTION_DIR={out}")
    print(f"POSES={poses.shape} START={poses[0,:3,3].tolist()} END={poses[-1,:3,3].tolist()}")


if __name__ == "__main__":
    main()
