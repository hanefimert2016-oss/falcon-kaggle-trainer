from __future__ import annotations

from dataclasses import dataclass
import math

import pybullet as p
import pybullet_data

from .config import FactoryConfig
from .layout import FactoryLayout, build_factory_layout


SIM_HZ = 240
JOINT_COUNT = 7
MAX_REACH_M = 1.45
SHOWCASE_PARTS = 6
PART_START_INTERVAL_S = 0.5
SEGMENT_DURATION_S = 1.5

ROUTE = (
    ("input", (-8.0, -2.3, 0.72)),
    ("pick_place", (-4.5, -2.3, 0.72)),
    ("assembly", (0.0, 0.0, 0.78)),
    ("joining", (3.0, 0.0, 0.78)),
    ("quality_control", (5.5, -1.25, 0.72)),
    ("packaging", (6.6, 1.0, 0.72)),
    ("output", (8.5, 2.3, 0.72)),
)
ROUTE_DURATION_S = (len(ROUTE) - 1) * SEGMENT_DURATION_S


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
class WorkpieceFrameState:
    position: tuple[float, float, float]
    stage: str
    owner_robot_id: str | None
    state: str


@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    time_s: float
    joint_positions: dict[str, tuple[float, ...]]
    workpieces: dict[str, WorkpieceFrameState]


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

        for robot in self.layout.robots:
            quat = p.getQuaternionFromEuler([0.0, 0.0, robot.base_yaw])
            body = p.loadURDF(
                "kuka_iiwa/model.urdf",
                basePosition=[
                    robot.base_position[0],
                    robot.base_position[1],
                    robot.base_position[2],
                ],
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
            p.getJointState(body, joint, physicsClientId=self.client_id)[0]
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
        for index, robot in enumerate(self.layout.robots):
            body = self.robot_bodies[robot.robot_id]
            phase = index * 0.37
            targets = (
                0.35 * math.sin(sim_time * 0.8 + phase),
                0.45 * math.sin(sim_time * 0.65 + phase + 0.4),
                0.40 * math.sin(sim_time * 0.70 + phase + 0.8),
                0.55 * math.sin(sim_time * 0.60 + phase + 1.2),
                0.30 * math.sin(sim_time * 0.90 + phase + 1.6),
                0.35 * math.sin(sim_time * 0.75 + phase + 2.0),
                0.25 * math.sin(sim_time * 1.00 + phase + 2.4),
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
    def _owner_for_stage(stage: str, part_index: int) -> str | None:
        pools = {
            "pick_place": tuple(f"R{i:02d}" for i in range(1, 9)),
            "assembly": tuple(f"R{i:02d}" for i in range(9, 13)),
            "joining": tuple(f"R{i:02d}" for i in range(13, 17)),
            "quality_control": ("R17", "R18"),
            "packaging": ("R19", "R20"),
        }
        pool = pools.get(stage)
        if not pool:
            return None
        return pool[part_index % len(pool)]

    def _workpiece_state(self, part_index: int, sim_time: float) -> WorkpieceFrameState:
        start_time = part_index * PART_START_INTERVAL_S
        if sim_time < start_time:
            return WorkpieceFrameState(
                position=ROUTE[0][1],
                stage="input",
                owner_robot_id=None,
                state="queued",
            )

        elapsed = sim_time - start_time
        if elapsed >= ROUTE_DURATION_S:
            return WorkpieceFrameState(
                position=ROUTE[-1][1],
                stage="output",
                owner_robot_id=None,
                state="complete",
            )

        segment = min(int(elapsed // SEGMENT_DURATION_S), len(ROUTE) - 2)
        local = (elapsed - segment * SEGMENT_DURATION_S) / SEGMENT_DURATION_S
        start_pos = ROUTE[segment][1]
        end_stage, end_pos = ROUTE[segment + 1]
        position = tuple(
            float(start_pos[i] * (1.0 - local) + end_pos[i] * local)
            for i in range(3)
        )
        return WorkpieceFrameState(
            position=position,
            stage=end_stage,
            owner_robot_id=self._owner_for_stage(end_stage, part_index),
            state="processing",
        )

    def _sample_workpieces(self, sim_time: float) -> dict[str, WorkpieceFrameState]:
        return {
            f"P{index + 1:03d}": self._workpiece_state(index, sim_time)
            for index in range(SHOWCASE_PARTS)
        }

    def run(self) -> SimulationResult:
        total_steps = max(1, int(round(self.config.duration_s * SIM_HZ)))
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
                self.frame_samples.append(
                    FrameSample(
                        frame_index=frame_index,
                        time_s=sim_time,
                        joint_positions=joints,
                        workpieces=self._sample_workpieces(sim_time),
                    )
                )
                for robot_id in sorted(self.robot_bodies):
                    spec = self._robot_specs[robot_id]
                    self.telemetry.append(
                        TelemetryEvent(
                            time_s=sim_time,
                            robot_id=robot_id,
                            phase="showcase_motion",
                            result="sampled",
                            cell_id=spec.cell_id,
                        )
                    )

        completed_parts = sum(
            1
            for index in range(SHOWCASE_PARTS)
            if self.config.duration_s >= index * PART_START_INTERVAL_S + ROUTE_DURATION_S
        )
        return SimulationResult(
            config=self.config,
            robot_count=self.robot_count,
            fixed_base_robot_count=self.fixed_base_robot_count,
            frame_samples=tuple(self.frame_samples),
            telemetry=tuple(self.telemetry),
            completed_parts=completed_parts,
            lost_parts=0,
            transfer_conflicts=0,
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
