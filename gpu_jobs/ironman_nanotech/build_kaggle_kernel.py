#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

BLENDER_URL = "https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz"
SLUG = "ironman-nanotech-3d"

KERNEL_TEMPLATE = r'''# Auto-generated Kaggle GPU job for Iron Man nanotech 3D generation.
import base64, json, os, pathlib, shutil, subprocess, tarfile, urllib.request

OUT = pathlib.Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

print("=== GPU ===", flush=True)
subprocess.run(["nvidia-smi"], check=False)

blender_url = __BLENDER_URL__
archive = OUT / "blender.tar.xz"
blender_dir = OUT / "blender"

if not blender_dir.exists():
    print("Downloading Blender:", blender_url, flush=True)
    urllib.request.urlretrieve(blender_url, archive)
    with tarfile.open(archive, "r:xz") as tf:
        tf.extractall(OUT)
    extracted = sorted(OUT.glob("blender-4.5.*-linux-x64"))
    if not extracted:
        raise RuntimeError("Blender archive extracted but executable directory was not found")
    extracted[0].rename(blender_dir)

blender = blender_dir / "blender"
if not blender.exists():
    raise RuntimeError("Blender executable missing: " + str(blender))
blender.chmod(0o755)

scene_py = OUT / "ironman_nanotech_scene.py"
scene_py.write_bytes(base64.b64decode(__SCENE_B64__))

env = os.environ.copy()
env["IRONMAN_OUT"] = str(OUT)
cmd = [str(blender), "-b", "--python", str(scene_py)]
print("+", " ".join(cmd), flush=True)
subprocess.check_call(cmd, env=env)

expected = [
    "ironman_nanotech_scene.blend",
    "ironman_nanotech_animated.glb",
    "ironman_nanotech_poster.png",
    "ironman_nanotech_transform.mp4",
    "ironman_nanotech_manifest.json",
]
sizes = {}
for name in expected:
    p = OUT / name
    if not p.exists():
        raise RuntimeError("missing output " + name)
    sizes[name] = p.stat().st_size

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     text=True, capture_output=True, check=False).stdout.strip()
summary = {"gpu": gpu, "files": sizes, "blender_url": blender_url}
(OUT / "kaggle_gpu_report.json").write_text(json.dumps(summary, indent=2) + "\n")
print("KAGGLE_3D_JOB_DONE", json.dumps(summary), flush=True)
'''

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--out", default="kernel_ironman_nanotech")
    ns = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    scene_path = root / "gpu_jobs" / "ironman_nanotech" / "blender_scene.py"
    scene = scene_path.read_bytes()
    kernel = KERNEL_TEMPLATE.replace("__BLENDER_URL__", repr(BLENDER_URL))
    kernel = kernel.replace("__SCENE_B64__", repr(base64.b64encode(scene).decode("ascii")))

    out = root / ns.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(kernel, encoding="utf-8")

    meta = {
        "id": f"{ns.owner}/{SLUG}",
        "title": "Ironman Nanotech 3D",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kernel": meta["id"], "scene_bytes": len(scene), "out": str(out)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
