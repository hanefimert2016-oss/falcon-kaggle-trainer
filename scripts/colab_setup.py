#!/usr/bin/env python3
"""Helpers for preparing Falcon FLM inside a Google Colab runtime."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch


def in_colab() -> bool:
    return (
        "google.colab" in sys.modules
        or bool(os.environ.get("COLAB_RELEASE_TAG"))
        or Path("/content").exists()
    )


def prepare(
    drive_root: str | Path = "/content/drive/MyDrive/FalconFLM",
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    drive_root = Path(drive_root).expanduser()
    data_path = Path(data_dir).expanduser() if data_dir else drive_root / "data"
    output_path = Path(output_dir).expanduser() if output_dir else drive_root / "runs" / "colab-v0.3"

    data_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    env = {
        "FLM_DATA_DIR": str(data_path),
        "FLM_OUTPUT_DIR": str(output_path),
    }
    for key, value in env.items():
        os.environ[key] = value

    info = {
        "in_colab": in_colab(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "drive_root": str(drive_root),
        "data_dir": str(data_path),
        "output_dir": str(output_path),
        "env": env,
    }
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Falcon FLM paths for Colab")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/FalconFLM")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the supplied paths without requiring an actual Colab runtime.",
    )
    args = parser.parse_args()

    info = prepare(args.drive_root, args.data_dir, args.output_dir)
    if not args.dry_run and not info["in_colab"]:
        raise SystemExit("This command is intended for Google Colab; use --dry-run for CI.")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
