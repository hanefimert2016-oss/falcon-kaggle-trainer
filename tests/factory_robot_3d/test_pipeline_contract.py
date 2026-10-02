from __future__ import annotations

from pathlib import Path

import pytest


REQUIRED = (
    "factory_robot_cinematic.mp4",
    "factory_config.json",
    "animation.json",
    "telemetry.csv",
    "summary.json",
    "final_frame.png",
    "preview_wide.png",
    "preview_close.png",
    "render.log",
)


def _write_valid_sized_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        (root / name).write_bytes(b"x")


def test_artifact_validator_accepts_complete_nonempty_output(tmp_path):
    from factory_robot_3d.pipeline.validate_artifacts import validate_artifacts

    _write_valid_sized_files(tmp_path)
    report = validate_artifacts(tmp_path)

    assert report.ok is True
    assert report.errors == ()
    assert tuple(path.name for path in report.files) == REQUIRED


def test_artifact_validator_rejects_missing_or_empty_file(tmp_path):
    from factory_robot_3d.pipeline.validate_artifacts import validate_artifacts

    _write_valid_sized_files(tmp_path)
    (tmp_path / "render.log").write_bytes(b"")
    (tmp_path / "preview_close.png").unlink()

    report = validate_artifacts(tmp_path)

    assert report.ok is False
    assert any("render.log" in error and "empty" in error for error in report.errors)
    assert any("preview_close.png" in error and "missing" in error for error in report.errors)


def test_production_steps_run_in_required_order():
    from factory_robot_3d.pipeline.run_production import PRODUCTION_STAGES

    assert PRODUCTION_STAGES == (
        "simulate",
        "build_scene",
        "preview_validate",
        "render",
        "encode",
        "validate",
    )
