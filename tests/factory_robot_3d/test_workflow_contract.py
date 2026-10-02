from __future__ import annotations

import json
from pathlib import Path


PRIMARY = Path(".github/workflows/factory-robot-3d.yml")
MANUAL = Path(".github/workflows/factory-robot-3d-kaggle.yml")
STATUS = Path(".github/workflows/factory-robot-3d-status.yml")
TEMPLATE = Path("factory_robot_3d/kernel-metadata.template.json")


def test_primary_workflow_owns_one_cinematic_t4_slug_and_full_package():
    text = PRIMARY.read_text(encoding="utf-8")

    assert "factory-robot-cinematic-t4" in text
    assert "NvidiaTeslaT4" in text
    assert "factory_robot_3d/**" in text
    assert "pytest tests/factory_robot_3d" in text
    assert "cp -R factory_robot_3d" in text
    assert "kaggle_entrypoint.py" in text
    assert "factory-robot-cinematic-t4-output" in text
    assert "factory-robot-3d-t4-simulation" not in text


def test_manual_duplicate_workflow_remains_dispatch_only():
    text = MANUAL.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "\n  push:" not in text
    assert "\n  pull_request:" not in text


def test_primary_wait_loop_distinguishes_quota_from_kernel_failure():
    text = PRIMARY.read_text(encoding="utf-8")

    assert "Maximum batch GPU session count" in text
    assert "quota_error" in text
    assert "kernel_error" in text
    assert "kaggle kernels logs" in text


def test_status_probe_uses_same_production_slug_without_push():
    text = STATUS.read_text(encoding="utf-8")

    assert "factory-robot-cinematic-t4" in text
    assert "factory-robot-3d-t4-simulation" not in text
    assert "kaggle kernels status" in text
    assert "kaggle kernels logs" in text


def test_kernel_template_requests_private_t4_gpu_production():
    data = json.loads(TEMPLATE.read_text(encoding="utf-8"))

    assert data["id"].endswith("/factory-robot-cinematic-t4")
    assert data["code_file"] == "kaggle_entrypoint.py"
    assert data["kernel_type"] == "script"
    assert data["is_private"] is True
    assert data["enable_gpu"] is True
    assert data["enable_internet"] is True
    assert data["machine_shape"] == "NvidiaTeslaT4"


def test_primary_downloads_revalidates_and_keeps_artifact_at_least_seven_days():
    text = PRIMARY.read_text(encoding="utf-8")

    assert "kaggle kernels output" in text
    assert "factory_robot_3d.pipeline.validate_video" in text
    assert "validate_artifacts" in text
    assert "retention-days:" in text

    retention_line = next(
        line for line in text.splitlines()
        if "retention-days:" in line
    )
    retention = int(retention_line.split(":", 1)[1].strip())
    assert retention >= 7
