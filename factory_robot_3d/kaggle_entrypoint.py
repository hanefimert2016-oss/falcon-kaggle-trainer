from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
OUTPUT = Path("/kaggle/working/factory_robot_cinematic_output")


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def _ensure_runtime() -> None:
    missing: list[str] = []
    try:
        import pybullet  # noqa: F401
    except Exception:
        missing.append("pybullet")
    if missing:
        _run([sys.executable, "-m", "pip", "install", "-q", *missing])

    for binary in ("nvidia-smi", "ffmpeg", "ffprobe", "bash"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"required runtime executable is missing: {binary}")


def _verify_t4() -> tuple[str, ...]:
    from factory_robot_3d.pipeline.kaggle_preflight import verify_gpu_identity

    result = _run(["nvidia-smi", "-L"], capture=True)
    print(result.stdout, flush=True)
    names = verify_gpu_identity(result.stdout)
    print("KAGGLE_T4_OK", ",".join(names), flush=True)
    return names


def _install_blender() -> Path:
    installer = ROOT / "factory_robot_3d" / "pipeline" / "install_blender.sh"
    result = _run(["bash", str(installer)], capture=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)

    candidates = [
        Path(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().endswith("/blender")
    ]
    if not candidates:
        raise RuntimeError("Blender installer did not report a blender binary")
    blender = candidates[-1]
    if not blender.is_file():
        raise RuntimeError(f"Blender binary does not exist after install: {blender}")

    version = _run([str(blender), "--version"], capture=True)
    first = version.stdout.splitlines()[0] if version.stdout else ""
    if first.strip() != "Blender 5.2.2":
        raise RuntimeError(f"expected Blender 5.2.2, got {first!r}")
    print("BLENDER_VERSION_OK", first, flush=True)
    return blender


def main() -> int:
    os.chdir(ROOT)
    _ensure_runtime()
    gpu_names = _verify_t4()
    blender = _install_blender()

    from factory_robot_3d.pipeline.run_production import run_production

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary = run_production(
        OUTPUT,
        blender_bin=blender,
        python_executable=sys.executable,
    )
    payload = {
        "success": bool(summary.artifacts.ok),
        "stages": list(summary.stages),
        "gpu_names": list(gpu_names),
        "blender": str(blender),
        "output_dir": str(summary.output_dir),
        "artifact_files": [path.name for path in summary.artifacts.files],
    }
    print("FACTORY_PRODUCTION_COMPLETE", json.dumps(payload, sort_keys=True), flush=True)
    if not summary.artifacts.ok:
        raise SystemExit("production artifact contract failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
