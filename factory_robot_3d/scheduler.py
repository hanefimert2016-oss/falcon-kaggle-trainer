from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .layout import FactoryLayout
from .models import TaskState, TransferZoneState, WorkpieceState


PRODUCTION_STAGES = (
    "pick_place",
    "assembly",
    "joining",
    "quality_control",
    "packaging",
    "output",
)


@dataclass
class FactoryWorldState:
    workpieces: dict[str, WorkpieceState] = field(default_factory=dict)
    tasks: dict[str, TaskState] = field(default_factory=dict)
    zones: dict[str, TransferZoneState] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotCommand:
    robot_id: str
    task_id: str
    workpiece_id: str
    stage: str


class FactoryScheduler:
    def __init__(self, layout: FactoryLayout, seed: int) -> None:
        self.layout = layout
        self.seed = seed
        self._workpiece_owners: dict[str, str] = {}
        self._zone_owners: dict[str, str] = {}

    def acquire_workpiece(self, workpiece_id: str, robot_id: str) -> bool:
        owner = self._workpiece_owners.get(workpiece_id)
        if owner is not None and owner != robot_id:
            return False
        self._workpiece_owners[workpiece_id] = robot_id
        return True

    def release_workpiece(self, workpiece_id: str, robot_id: str) -> bool:
        if self._workpiece_owners.get(workpiece_id) != robot_id:
            return False
        del self._workpiece_owners[workpiece_id]
        return True

    def acquire_zone(self, zone_id: str, robot_id: str) -> bool:
        owner = self._zone_owners.get(zone_id)
        if owner is not None and owner != robot_id:
            return False
        self._zone_owners[zone_id] = robot_id
        return True

    def release_zone(self, zone_id: str, robot_id: str) -> bool:
        if self._zone_owners.get(zone_id) != robot_id:
            return False
        del self._zone_owners[zone_id]
        return True

    def advance_workpiece_stage(self, workpiece: WorkpieceState, target_stage: str) -> None:
        try:
            current_index = PRODUCTION_STAGES.index(workpiece.stage)
            target_index = PRODUCTION_STAGES.index(target_stage)
        except ValueError as exc:
            raise ValueError("unknown production stage") from exc
        if target_index != current_index + 1:
            raise ValueError(f"cannot skip production stage from {workpiece.stage} to {target_stage}")
        workpiece.stage = target_stage

    def step(self, sim_time: float, world: FactoryWorldState) -> list[RobotCommand]:
        del sim_time  # ordering is deterministic and time-independent until ready-times are introduced.
        commands: list[RobotCommand] = []
        busy_robots: set[str] = set()

        stage_order = {stage: index for index, stage in enumerate(PRODUCTION_STAGES)}
        tasks = sorted(
            world.tasks.values(),
            key=lambda task: (stage_order.get(task.stage, len(stage_order)), task.task_id),
        )

        robots_by_role: dict[str, list[str]] = {}
        for robot in sorted(self.layout.robots, key=lambda item: item.robot_id):
            robots_by_role.setdefault(robot.role, []).append(robot.robot_id)

        for task in tasks:
            if task.owner_robot_id is not None:
                continue
            candidates = robots_by_role.get(task.stage, [])
            robot_id = next((rid for rid in candidates if rid not in busy_robots), None)
            if robot_id is None:
                continue
            if not self.acquire_workpiece(task.workpiece_id, robot_id):
                continue

            busy_robots.add(robot_id)
            task.owner_robot_id = robot_id
            workpiece = world.workpieces.get(task.workpiece_id)
            if workpiece is not None:
                workpiece.owner_robot_id = robot_id

            commands.append(
                RobotCommand(
                    robot_id=robot_id,
                    task_id=task.task_id,
                    workpiece_id=task.workpiece_id,
                    stage=task.stage,
                )
            )

        return commands
