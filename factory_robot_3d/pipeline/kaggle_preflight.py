from __future__ import annotations

from enum import Enum
import re


class KaggleStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    KERNEL_ERROR = "kernel_error"
    QUOTA_ERROR = "quota_error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


def classify_kaggle_status(text: str) -> KaggleStatus:
    value = text.upper()
    if "MAXIMUM BATCH GPU SESSION COUNT" in value or "GPU SESSION COUNT" in value:
        return KaggleStatus.QUOTA_ERROR
    if "CANCEL" in value:
        return KaggleStatus.CANCELLED
    if "COMPLETE" in value or "FINISHED" in value:
        return KaggleStatus.COMPLETE
    if "RUNNING" in value:
        return KaggleStatus.RUNNING
    if "QUEUED" in value or "PENDING" in value:
        return KaggleStatus.QUEUED
    if "ERROR" in value or "FAILED" in value:
        return KaggleStatus.KERNEL_ERROR
    return KaggleStatus.UNKNOWN


def verify_gpu_identity(nvidia_smi_text: str) -> tuple[str, ...]:
    names: list[str] = []
    for line in nvidia_smi_text.splitlines():
        match = re.search(r"GPU\s+\d+:\s+(.+?)(?:\s+\(UUID:|$)", line.strip())
        if match:
            name = match.group(1).strip()
            if "TESLA T4" in name.upper():
                names.append("Tesla T4")
    if not names:
        raise RuntimeError("Kaggle production requires at least one Tesla T4 GPU")
    return tuple(names)
