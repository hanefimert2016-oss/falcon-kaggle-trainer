from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

from .simulation import SimulationResult


@dataclass(frozen=True)
class ArtifactManifest:
    output_dir: Path
    factory_config_json: Path
    animation_json: Path
    telemetry_csv: Path
    summary_json: Path


def _round_float(value: float) -> float:
    return round(float(value), 9)


def _config_payload(result: SimulationResult) -> dict:
    config = result.config
    return {
        "robot_count": config.robot_count,
        "cell_count": config.cell_count,
        "fps": config.fps,
        "duration_s": config.duration_s,
        "frame_count": config.frame_count,
        "seed": config.seed,
        "role_counts": dict(sorted(config.role_counts.items())),
    }


def _serialize_workpiece_state(state: object) -> dict:
    if isinstance(state, dict):
        position = state.get("position", (0.0, 0.0, 0.0))
        return {
            "position": [_round_float(x) for x in position],
            "stage": state.get("stage", "unknown"),
            "owner_robot_id": state.get("owner_robot_id"),
            "state": state.get("state", "unknown"),
        }
    return {
        "position": [_round_float(x) for x in state.position],
        "stage": state.stage,
        "owner_robot_id": getattr(state, "owner_robot_id", None),
        "state": state.state,
    }


def _animation_payload(result: SimulationResult) -> dict:
    frames = []
    for sample in result.frame_samples:
        robots = {
            robot_id: [_round_float(x) for x in sample.joint_positions[robot_id]]
            for robot_id in sorted(sample.joint_positions)
        }
        workpieces = {
            workpiece_id: _serialize_workpiece_state(state)
            for workpiece_id, state in sorted(sample.workpieces.items())
        }
        frames.append(
            {
                "frame_index": sample.frame_index,
                "time_s": _round_float(sample.time_s),
                "robots": robots,
                "workpieces": workpieces,
                "conveyor_offsets": {
                    "MAIN_IN": _round_float(sample.time_s * 0.35),
                    "TRANSFER": _round_float(sample.time_s * 0.25),
                    "MAIN_OUT": _round_float(sample.time_s * 0.30),
                },
                "warning_lights": {
                    f"C{i}": bool((sample.frame_index // 24 + i) % 2)
                    for i in range(1, 6)
                },
                "active_task_ids": [
                    f"T-{wid}-{state.stage}"
                    for wid, state in sorted(sample.workpieces.items())
                    if state.state == "processing"
                ],
            }
        )
    return {"schema_version": 1, "frames": frames}


def _summary_payload(result: SimulationResult) -> dict:
    full_showcase = (
        result.config.robot_count == 20
        and result.config.cell_count == 5
        and result.config.frame_count == 288
    )
    success = (
        result.lost_parts == 0
        and result.transfer_conflicts == 0
        and len(result.frame_samples) == result.config.frame_count
        and (not full_showcase or result.completed_parts >= 6)
    )
    return {
        "success": success,
        "robot_count": result.robot_count,
        "cell_count": result.config.cell_count,
        "completed_parts": result.completed_parts,
        "lost_parts": result.lost_parts,
        "transfer_conflicts": result.transfer_conflicts,
        "simulation_frames": len(result.frame_samples),
        "fps": result.config.fps,
        "duration_s": result.config.duration_s,
        "seed": result.config.seed,
        "animation": "animation.json",
        "telemetry": "telemetry.csv",
    }


def write_simulation_artifacts(result: SimulationResult, output_dir: Path) -> ArtifactManifest:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factory_config_json = output_dir / "factory_config.json"
    animation_json = output_dir / "animation.json"
    telemetry_csv = output_dir / "telemetry.csv"
    summary_json = output_dir / "summary.json"

    factory_config_json.write_text(
        json.dumps(_config_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    animation_json.write_text(
        json.dumps(_animation_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with telemetry_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "robot_id",
                "time_s",
                "cell_id",
                "task_id",
                "workpiece_id",
                "phase",
                "transfer_zone_id",
                "attached",
                "result",
            ]
        )
        for event in result.telemetry:
            writer.writerow(
                [
                    event.robot_id,
                    f"{event.time_s:.9f}",
                    event.cell_id,
                    event.task_id,
                    event.workpiece_id,
                    event.phase,
                    event.transfer_zone_id,
                    int(event.attached),
                    event.result,
                ]
            )

    summary_json.write_text(
        json.dumps(_summary_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return ArtifactManifest(
        output_dir=output_dir,
        factory_config_json=factory_config_json,
        animation_json=animation_json,
        telemetry_csv=telemetry_csv,
        summary_json=summary_json,
    )
