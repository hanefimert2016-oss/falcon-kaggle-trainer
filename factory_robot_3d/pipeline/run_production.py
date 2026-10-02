from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from typing import Sequence

from .validate_artifacts import ArtifactValidationReport, validate_artifacts


PRODUCTION_STAGES = (
    "simulate",
    "build_scene",
    "preview_validate",
    "render",
    "encode",
    "validate",
)


@dataclass(frozen=True)
class ProductionSummary:
    output_dir: Path
    stages: tuple[str, ...]
    artifacts: ArtifactValidationReport


def _run(command: Sequence[str], *, log=None, env=None) -> None:
    subprocess.run(
        list(command),
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT if log is not None else None,
        env=env,
    )


def run_production(
    output_dir: Path,
    *,
    blender_bin: Path | None = None,
    python_executable: str = sys.executable,
) -> ProductionSummary:
    """Execute the complete local/Kaggle production chain.

    Frame rendering and encoding are delegated to the project shell wrappers;
    this keeps orchestration deterministic and lets the same entry point run
    in Kaggle and smoke environments.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blender_bin = Path(blender_bin or os.environ.get("BLENDER_BIN", "blender"))
    scene_path = output_dir / "factory_robot_cinematic.blend"
    render_log = output_dir / "render.log"

    _run([
        python_executable,
        "-m",
        "factory_robot_3d.run_simulation",
        "--output",
        str(output_dir),
    ])
    simulation_summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    if not simulation_summary.get("success"):
        raise RuntimeError("simulation failed success criteria; render not started")

    with render_log.open("w", encoding="utf-8") as log:
        _run([
            str(blender_bin),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python",
            "factory_robot_3d/blender/probe_gpu.py",
        ], log=log)

        _run([
            str(blender_bin),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python",
            "factory_robot_3d/blender/build_scene.py",
            "--",
            "--input",
            str(output_dir),
            "--output",
            str(scene_path),
        ], log=log)

        _run([
            "bash",
            "factory_robot_3d/pipeline/render_frames.sh",
            str(blender_bin),
            str(scene_path),
            str(output_dir),
        ], log=log)

        _run([
            "bash",
            "factory_robot_3d/pipeline/encode_video.sh",
            str(output_dir),
        ], log=log)

    _run([
        python_executable,
        "-m",
        "factory_robot_3d.pipeline.validate_video",
        str(output_dir / "factory_robot_cinematic.mp4"),
    ])

    report = validate_artifacts(output_dir)
    if not report.ok:
        raise RuntimeError("artifact validation failed: " + "; ".join(report.errors))

    frames_dir = output_dir / "frames"
    if frames_dir.is_dir():
        shutil.rmtree(frames_dir)

    return ProductionSummary(output_dir=output_dir, stages=PRODUCTION_STAGES, artifacts=report)
