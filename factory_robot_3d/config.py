from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


APPROVED_ROLE_COUNTS = {
    "pick_place": 8,
    "assembly": 4,
    "joining": 4,
    "quality_control": 2,
    "packaging": 2,
}


@dataclass(frozen=True)
class FactoryConfig:
    robot_count: int
    cell_count: int
    fps: int
    duration_s: float
    seed: int
    role_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return int(round(self.fps * self.duration_s))

    @classmethod
    def cinematic_default(cls) -> "FactoryConfig":
        return cls(
            robot_count=20,
            cell_count=5,
            fps=24,
            duration_s=12.0,
            seed=20261002,
            role_counts=dict(APPROVED_ROLE_COUNTS),
        )

    def validate(self) -> None:
        if self.robot_count <= 0:
            raise ValueError("robot_count must be positive")
        if self.cell_count <= 0:
            raise ValueError("cell_count must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if sum(self.role_counts.values()) != self.robot_count:
            raise ValueError("role counts must sum to robot_count")
        if any(count < 0 for count in self.role_counts.values()):
            raise ValueError("role counts must be non-negative")
