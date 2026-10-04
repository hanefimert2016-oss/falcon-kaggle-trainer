from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


_REQUIRED_SUMMARY_FIELDS = {
    "success",
    "robot_count",
    "cell_count",
    "completed_parts",
    "lost_parts",
    "transfer_conflicts",
    "simulation_frames",
    "fps",
    "duration_s",
    "seed",
}


@dataclass(frozen=True)
class AnimationBundle:
    config: Mapping[str, Any]
    frames: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    robot_ids: tuple[str, ...]

    @property
    def robot_count(self) -> int:
        return len(self.robot_ids)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(int(frame["frame_index"]) for frame in self.frames)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing animation bundle file: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path.name}") from exc


def _validate_joint_vector(robot_id: str, values: object, frame_index: int) -> None:
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError(
            f"robot {robot_id} frame {frame_index} must contain seven joint values"
        )
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        raise ValueError(
            f"robot {robot_id} frame {frame_index} joint values must be finite"
        )


def load_animation_bundle(input_dir: Path) -> AnimationBundle:
    input_dir = Path(input_dir)
    config = _read_json(input_dir / "factory_config.json")
    animation = _read_json(input_dir / "animation.json")
    summary = _read_json(input_dir / "summary.json")

    if not isinstance(config, dict):
        raise ValueError("factory config must be an object")
    if int(config.get("robot_count", -1)) != 20:
        raise ValueError("cinematic animation bundle requires exactly 20 robots")
    if int(config.get("cell_count", -1)) != 5:
        raise ValueError("cinematic animation bundle requires exactly 5 cells")
    if int(config.get("frame_count", -1)) != 288:
        raise ValueError("cinematic animation bundle requires exactly 288 frames")
    if int(config.get("fps", -1)) != 24:
        raise ValueError("cinematic animation bundle requires 24 FPS")

    if not isinstance(summary, dict) or not _REQUIRED_SUMMARY_FIELDS.issubset(summary):
        missing = sorted(_REQUIRED_SUMMARY_FIELDS - set(summary if isinstance(summary, dict) else ()))
        raise ValueError(f"simulation summary missing required fields: {missing}")
    if int(summary.get("robot_count", -1)) != 20:
        raise ValueError("simulation summary must report 20 robots")
    if int(summary.get("simulation_frames", -1)) != 288:
        raise ValueError("simulation summary must report 288 frames")

    if not isinstance(animation, dict) or not isinstance(animation.get("frames"), list):
        raise ValueError("animation.json must contain a frames array")
    raw_frames = animation["frames"]
    if len(raw_frames) != 288:
        raise ValueError(f"animation must contain exactly 288 frames, got {len(raw_frames)}")

    expected_robot_ids = tuple(f"R{i:02d}" for i in range(1, 21))
    expected_robot_set = set(expected_robot_ids)
    expected_indices = list(range(1, 289))
    actual_indices: list[int] = []

    for frame in raw_frames:
        if not isinstance(frame, dict):
            raise ValueError("every animation frame must be an object")
        frame_index = frame.get("frame_index")
        if not isinstance(frame_index, int):
            raise ValueError("animation frame index must be an integer")
        actual_indices.append(frame_index)

        robots = frame.get("robots")
        if not isinstance(robots, dict) or set(robots) != expected_robot_set:
            raise ValueError(
                f"frame {frame_index} must contain all 20 robot IDs exactly once"
            )
        for robot_id in expected_robot_ids:
            _validate_joint_vector(robot_id, robots[robot_id], frame_index)

    if actual_indices != expected_indices:
        raise ValueError("animation frame indices must be contiguous 1..288 with no duplicates")

    return AnimationBundle(
        config=config,
        frames=tuple(raw_frames),
        summary=summary,
        robot_ids=expected_robot_ids,
    )
