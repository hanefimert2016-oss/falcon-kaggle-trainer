from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import pybullet as p
import pybullet_data

from .config import FactoryConfig
from .layout import FactoryLayout, build_factory_layout
from .scheduler import FactoryScheduler


SIM_HZ = 240
JOINT_COUNT = 7
MAX_REACH_M = 1.45
PART_COUNT = 8
PART_INTERVAL_S = 0.75
PART_CYCLE_S = 6.0

PRODUCTION_WAYPOINTS = (
    ("pick_place", (-8.0, -2.3, 0.72)),
    ("assembly", (-1.2, 0.0, 0.78)),
    ("joining", (2.2, 0.0, 0.78)),
    ("quality_control", (5.45, -1.15, 0.74)),
    ("packaging", (6.65, 1.0, 0.74)),
    ("output", (8.35, 2.3, 0.72)),
    ("complete", (9.0, 2.3, 0.72)),
)


@dataclass(frozen=True)
class TelemetryEvent:
    time_s: float
    robot_id: str
    phase: str
    result: str
    cell_id: str = ""
    task_id: str = ""
    workpiece_id: str = ""
    transfer_zone_id: str = ""
    attached: bool = False


@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    time_s: float
    joint_positions: Mapping[str, tuple[float, ...]]
    workpieces: Mapping[str, dict[str, object]]


@dataclass(frozen=True)
class SimulationResult:
    config: FactoryConfig
    robot_count: int
    fixed_base_robot_count: int
    frame_samples: tuple[FrameSample, ...]
    telemetry: tuple[TelemetryEvent, ...]
    completed_parts: int = 0
    lost_parts: int = 0
    transfer_conflicts: int = 0


class FactorySimulation:
    def __init__(self, config: FactoryConfig) -> None:
        config.validate()
        self.config = config
        self.layout: FactoryLayout = build_factory_layout(config)
        self.client_id = p.connect(p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("PyBullet DIRECT connection failed")

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            numSolverIterations=80,
            deterministicOverlappingPairs=1,
            physicsClientId=self.client_id,
        )
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        self.robot_bodies: dict[str, int] = {}
        self._robot_specs = {robot.robot_id: robot for robot in self.layout.robots}
        self.telemetry: list[TelemetryEvent] = []
        self.frame_samples: list[FrameSample] = []
        self._closed = False
        self.scheduler = FactoryScheduler(self.layout, seed=config.seed)
        self._workpiece_stages: dict[str, str] = {}
        self._workpiece_owners: dict[str, str] = {}
        self._transfer_conflicts = 0

        for robot in self.layout.robots:
            quat = p.getQuaternionFromEuler([0.0, 0.0, robot.base_yaw])
            body = p.loadURDF(
                "kuka_iiwa/model.urdf",
                basePosition=list(robot.base_position),
                baseOrientation=quat,
                useFixedBase=True,
                physicsClientId=self.client_id,
            )
            self.robot_bodies[robot.robot_id] = body
            for joint in range(JOINT_COUNT):
                p.changeDynamics(
                    body,
                    joint,
                    linearDamping=0.04,
                    angularDamping=0.04,
                    physicsClientId=self.client_id,
                )

    @property
    def robot_count(self) -> int:
        return len(self.robot_bodies)

    @property
    def fixed_base_robot_count(self) -> int:
        return len(self.robot_bodies)

    def get_joint_positions(self, robot_id: str) -> tuple[float, ...]:
        body = self.robot_bodies[robot_id]
        return tuple(
            float(p.getJointState(body, joint, physicsClientId=self.client_id)[0])
            for joint in range(JOINT_COUNT)
        )

    def command_robot_target(self, robot_id: str, target: tuple[float, float, float]) -> bool:
        spec = self._robot_specs[robot_id]
        dx = target[0] - spec.base_position[0]
        dy = target[1] - spec.base_position[1]
        dz = target[2] - spec.base_position[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > MAX_REACH_M or target[2] < 0.0:
            self.telemetry.append(
                TelemetryEvent(
                    time_s=0.0,
                    robot_id=robot_id,
                    phase="manual_target",
                    result="failed_unreachable",
                    cell_id=spec.cell_id,
                )
            )
            return False

        body = self.robot_bodies[robot_id]
        orientation = p.getQuaternionFromEuler([0.0, math.pi, 0.0])
        solution = p.calculateInverseKinematics(
            body,
            6,
            target,
            orientation,
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.client_id,
        )
        for joint, angle in enumerate(solution[:JOINT_COUNT]):
            p.setJointMotorControl2(
                body,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(angle),
                force=500.0,
                positionGain=0.18,
                velocityGain=0.9,
                physicsClientId=self.client_id,
            )
        return True

    def _apply_showcase_motion(self, sim_time: float) -> None:
        role_speed = {
            "pick_place": 1.15,
            "assembly": 0.82,
            "joining": 0.92,
            "quality_control": 0.68,
            "packaging": 1.05,
        }
        for index, robot in enumerate(self.layout.robots):
            body = self.robot_bodies[robot.robot_id]
            phase = index * 0.37
            speed = role_speed[robot.role]
            targets = (
                0.38 * math.sin(sim_time * 0.8 * speed + phase),
                0.34 + 0.34 * math.sin(sim_time * 0.65 * speed + phase + 0.4),
                -0.30 + 0.30 * math.sin(sim_time * 0.70 * speed + phase + 0.8),
                -0.82 + 0.42 * math.sin(sim_time * 0.60 * speed + phase + 1.2),
                0.24 * math.sin(sim_time * 0.90 * speed + phase + 1.6),
                0.48 + 0.28 * math.sin(sim_time * 0.75 * speed + phase + 2.0),
                0.32 * math.sin(sim_time * 1.00 * speed + phase + 2.4),
            )
            for joint, angle in enumerate(targets):
                p.setJointMotorControl2(
                    body,
                    joint,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=400.0,
                    positionGain=0.12,
                    velocityGain=0.8,
                    physicsClientId=self.client_id,
                )

    @staticmethod
    def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], u: float) -> list[float]:
        return [round(a[i] * (1.0 - u) + b[i] * u, 8) for i in range(3)]

    @staticmethod
    def _owner_for_stage(stage: str, part_index: int) -> str | None:
        pools = {
            "pick_place": tuple(f"R{i:02d}" for i in range(1, 9)),
            "assembly": tuple(f"R{i:02d}" for i in range(9, 13)),
            "joining": tuple(f"R{i:02d}" for i in range(13, 17)),
            "quality_control": ("R17", "R18"),
            "packaging": ("R19", "R20"),
        }
        pool = pools.get(stage)
        return None if not pool else pool[part_index % len(pool)]

    def _zone_for_transition(self, from_stage: str, to_stage: str):
        return next(
            (
                zone
                for zone in self.layout.transfer_zones
                if zone.from_stage == from_stage and zone.to_stage == to_stage
            ),
            None,
        )

    def _sync_workpiece_resources(
        self,
        workpieces: dict[str, dict[str, object]],
        sim_time: float,
    ) -> None:
        for workpiece_id, state in sorted(workpieces.items()):
            stage = str(state["stage"])
            part_index = int(workpiece_id[1:]) - 1
            previous_stage = self._workpiece_stages.get(workpiece_id)
            previous_owner = self._workpiece_owners.get(workpiece_id)
            next_owner = self._owner_for_stage(stage, part_index)

            if previous_owner and previous_owner != next_owner:
                self.scheduler.release_workpiece(workpiece_id, previous_owner)

            if next_owner and next_owner != previous_owner:
                if not self.scheduler.acquire_workpiece(workpiece_id, next_owner):
                    self._transfer_conflicts += 1
                    next_owner = None

            if previous_stage and previous_stage != stage:
                zone = self._zone_for_transition(previous_stage, stage)
                if zone is not None:
                    transfer_robot = next_owner or previous_owner or "SYSTEM"
                    acquired = self.scheduler.acquire_zone(zone.zone_id, transfer_robot)
                    if acquired:
                        self.scheduler.release_zone(zone.zone_id, transfer_robot)
                    else:
                        self._transfer_conflicts += 1
                    robot_spec = self._robot_specs.get(transfer_robot)
                    self.telemetry.append(
                        TelemetryEvent(
                            time_s=sim_time,
                            robot_id=transfer_robot,
                            phase="transfer",
                            result="transferred" if acquired else "transfer_conflict",
                            cell_id="" if robot_spec is None else robot_spec.cell_id,
                            task_id=f"T-{workpiece_id}-{stage}",
                            workpiece_id=workpiece_id,
                            transfer_zone_id=zone.zone_id,
                            attached=next_owner is not None,
                        )
                    )

            state["owner_robot_id"] = next_owner
            self._workpiece_stages[workpiece_id] = stage
            if next_owner:
                self._workpiece_owners[workpiece_id] = next_owner
            else:
                self._workpiece_owners.pop(workpiece_id, None)

    def _sample_workpieces(self, sim_time: float) -> dict[str, dict[str, object]]:
        sampled: dict[str, dict[str, object]] = {}
        for index in range(PART_COUNT):
            workpiece_id = f"P{index + 1:03d}"
            start = index * PART_INTERVAL_S
            age = sim_time - start
            if age < 0.0:
                continue

            if age >= PART_CYCLE_S:
                sampled[workpiece_id] = {
                    "stage": "complete",
                    "state": "complete",
                    "position": list(PRODUCTION_WAYPOINTS[-1][1]),
                }
                continue

            segment_length = PART_CYCLE_S / (len(PRODUCTION_WAYPOINTS) - 1)
            segment = min(int(age / segment_length), len(PRODUCTION_WAYPOINTS) - 2)
            local_u = (age - segment * segment_length) / segment_length
            stage = PRODUCTION_WAYPOINTS[segment][0]
            start_pos = PRODUCTION_WAYPOINTS[segment][1]
            end_pos = PRODUCTION_WAYPOINTS[segment + 1][1]
            sampled[workpiece_id] = {
                "stage": stage,
                "state": "active",
                "position": self._lerp(start_pos, end_pos, local_u),
            }
        return sampled

    def _completed_part_count(self, sim_time: float) -> int:
        return sum(
            1
            for index in range(PART_COUNT)
            if sim_time - index * PART_INTERVAL_S >= PART_CYCLE_S
        )

    def run(self) -> SimulationResult:
        total_steps = max(1, int(round(self.config.duration_s * SIM_HZ)))
        if SIM_HZ % self.config.fps != 0:
            raise ValueError("fps must divide 240 Hz simulation rate exactly")
        capture_every = SIM_HZ // self.config.fps
        frame_index = 0

        for step in range(1, total_steps + 1):
            sim_time = step / SIM_HZ
            self._apply_showcase_motion(sim_time)
            p.stepSimulation(physicsClientId=self.client_id)

            if step % capture_every == 0:
                frame_index += 1
                joints = {
                    robot_id: self.get_joint_positions(robot_id)
                    for robot_id in sorted(self.robot_bodies)
                }
                workpieces = self._sample_workpieces(sim_time)
                self._sync_workpiece_resources(workpieces, sim_time)
                self.frame_samples.append(
                    FrameSample(
                        frame_index=frame_index,
                        time_s=sim_time,
                        joint_positions=joints,
                        workpieces=workpieces,
                    )
                )

                for robot in sorted(self.layout.robots, key=lambda item: item.robot_id):
                    self.telemetry.append(
                        TelemetryEvent(
                            time_s=sim_time,
                            robot_id=robot.robot_id,
                            phase="showcase_motion",
                            result="sampled",
                            cell_id=robot.cell_id,
                        )
                    )

        final_time = total_steps / SIM_HZ
        return SimulationResult(
            config=self.config,
            robot_count=self.robot_count,
            fixed_base_robot_count=self.fixed_base_robot_count,
            frame_samples=tuple(self.frame_samples),
            telemetry=tuple(self.telemetry),
            completed_parts=self._completed_part_count(final_time),
            lost_parts=0,
            transfer_conflicts=self._transfer_conflicts,
        )

    def close(self) -> None:
        if not self._closed:
            p.disconnect(physicsClientId=self.client_id)
            self._closed = True


def run_simulation(config: FactoryConfig) -> SimulationResult:
    sim = FactorySimulation(config)
    try:
        return sim.run()
    finally:
        sim.close()
