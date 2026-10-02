from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

EMBEDDED_PROJECT_ZIP_B64 = "__EMBEDDED_PROJECT_ZIP_B64__"
EMBEDDED_SIMULATION_ZIP_B64 = "__EMBEDDED_SIMULATION_ZIP_B64__"
OUTPUT = Path("/kaggle/working/factory_robot_cinematic_output")


def _prepare_project_root() -> Path:
    root = Path("/kaggle/working/factory_robot_project")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(EMBEDDED_PROJECT_ZIP_B64)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        archive.extractall(root)
    sys.path.insert(0, str(root))
    print("EMBEDDED_PROJECT_READY", f"zip_bytes={len(payload)}", flush=True)
    return root


def _prepare_simulation_output() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(EMBEDDED_SIMULATION_ZIP_B64)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        archive.extractall(OUTPUT)
    required = ("factory_config.json", "animation.json", "telemetry.csv", "summary.json")
    for name in required:
        path = OUTPUT / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"precomputed simulation file missing: {name}")
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("success"):
        raise RuntimeError("precomputed simulation summary failed")
    print(
        "PRECOMPUTED_SIMULATION_READY",
        f"zip_bytes={len(payload)}",
        f"frames={summary.get('simulation_frames')}",
        f"robots={summary.get('robot_count')}",
        flush=True,
    )


def main() -> int:
    root = _prepare_project_root()
    os.chdir(root)
    _prepare_simulation_output()

    from factory_robot_3d.kaggle_entrypoint import _install_blender, _verify_t4
    from factory_robot_3d.pipeline.run_production import run_production

    for binary in ("nvidia-smi", "ffmpeg", "ffprobe", "bash"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"required runtime executable is missing: {binary}")

    gpu_names = _verify_t4()
    blender = _install_blender()
    summary = run_production(
        OUTPUT,
        blender_bin=blender,
        python_executable=sys.executable,
        skip_simulation=True,
    )
    payload = {
        "success": bool(summary.artifacts.ok),
        "stages": list(summary.stages),
        "gpu_names": list(gpu_names),
        "blender": str(blender),
        "output_dir": str(summary.output_dir),
        "artifact_files": [path.name for path in summary.artifacts.files],
        "precomputed": True,
    }
    print("FACTORY_PRODUCTION_COMPLETE", json.dumps(payload, sort_keys=True), flush=True)
    return 0 if summary.artifacts.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
