from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    cell_id: str
    role: str
    base_position: Vec3
    base_yaw: float


@dataclass
class WorkpieceState:
    workpiece_id: str
    stage: str
    owner_robot_id: Optional[str] = None


@dataclass
class TaskState:
    task_id: str
    stage: str
    workpiece_id: str
    owner_robot_id: Optional[str] = None


@dataclass
class TransferZoneState:
    zone_id: str
    owner_robot_id: Optional[str] = None
