from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


EMBEDDED_PROJECT_ZIP_B64 = "__EMBEDDED_PROJECT_ZIP_B64__"
OUTPUT = Path("/kaggle/working/factory_robot_cinematic_output")


def _prepare_project_root() -> Path:
    if EMBEDDED_PROJECT_ZIP_B64.startswith("__EMBEDDED_"):
        root = Path(__file__).resolve().parent.parent
    else:
        root = Path("/kaggle/working/factory_robot_project")
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        payload = base64.b64decode(EMBEDDED_PROJECT_ZIP_B64)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            archive.extractall(root)
        print(
            "EMBEDDED_PROJECT_READY",
            f"root={root}",
            f"zip_bytes={len(payload)}",
            flush=True,
        )

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


ROOT = _prepare_project_root()


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
    print("+ bash", installer, flush=True)
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\\n") else "\\n", flush=True)
    if result.stderr:
        print(
            result.stderr,
            end="" if result.stderr.endswith("\\n") else "\\n",
            file=sys.stderr,
            flush=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Blender installer failed with exit code {result.returncode}"
        )

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
