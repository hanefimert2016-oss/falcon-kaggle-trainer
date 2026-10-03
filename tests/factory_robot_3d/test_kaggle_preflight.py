from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("KernelWorkerStatus.QUEUED", "queued"),
        ("KernelWorkerStatus.RUNNING", "running"),
        ("KernelWorkerStatus.COMPLETE", "complete"),
        ("KernelWorkerStatus.ERROR", "kernel_error"),
        ("Maximum batch GPU session count of 2 reached", "quota_error"),
        ("KernelWorkerStatus.CANCELLED", "cancelled"),
    ],
)
def test_classify_kaggle_status(text, expected):
    from factory_robot_3d.pipeline.kaggle_preflight import classify_kaggle_status

    assert classify_kaggle_status(text).value == expected


def test_verify_gpu_identity_accepts_one_or_more_tesla_t4():
    from factory_robot_3d.pipeline.kaggle_preflight import verify_gpu_identity

    text = """GPU 0: Tesla T4 (UUID: GPU-a)
GPU 1: Tesla T4 (UUID: GPU-b)"""
    assert verify_gpu_identity(text) == ("Tesla T4", "Tesla T4")


@pytest.mark.parametrize("text", ["", "GPU 0: A100", "nvidia-smi unavailable"])
def test_verify_gpu_identity_rejects_missing_t4(text):
    from factory_robot_3d.pipeline.kaggle_preflight import verify_gpu_identity

    with pytest.raises(RuntimeError, match="T4"):
        verify_gpu_identity(text)
