#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
from typing import Sequence

ROOT = pathlib.Path(__file__).resolve().parent
KAGGLE_DIR = ROOT / "kaggle"
META_PATH = KAGGLE_DIR / "kernel-metadata.json"
DEFAULT_SLUG = "falcon-flm-train"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"


def _owner() -> str:
    value = (os.environ.get("KAGGLE_OWNER") or os.environ.get("KAGGLE_USERNAME") or "").strip()
    if not value:
        raise SystemExit("KAGGLE_OWNER is required (set it to your Kaggle username/owner slug)")
    return value


def _require_auth() -> None:
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return
    if os.environ.get("KAGGLE_USERNAME", "").strip() and os.environ.get("KAGGLE_KEY", "").strip():
        return
    raise SystemExit("Kaggle auth missing: set KAGGLE_API_TOKEN (preferred)")


def _kernel_ref(slug: str | None = None) -> str:
    actual_slug = slug or os.environ.get("KAGGLE_KERNEL_SLUG", DEFAULT_SLUG)
    return f"{_owner()}/{actual_slug}"


def render(slug: str, accelerator: str) -> pathlib.Path:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": _kernel_ref(slug),
        "title": slug.replace("-", " ").title(),
        "code_file": "train_flm.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": accelerator,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return META_PATH


def validate() -> None:
    train = KAGGLE_DIR / "train_flm.py"
    if not train.is_file():
        raise SystemExit(f"missing {train}")
    if not META_PATH.is_file():
        slug = os.environ.get("KAGGLE_KERNEL_SLUG", DEFAULT_SLUG)
        render(slug, DEFAULT_ACCELERATOR)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    required = ["id", "title", "code_file", "language", "kernel_type"]
    missing = [k for k in required if not meta.get(k)]
    if missing:
        raise SystemExit(f"metadata missing fields: {', '.join(missing)}")
    if meta["code_file"] != train.name:
        raise SystemExit("kernel metadata code_file does not match train_flm.py")
    if not meta.get("enable_gpu"):
        raise SystemExit("kernel metadata must enable GPU")
    print(f"OK metadata={META_PATH} kernel={meta['id']} accelerator={meta.get('machine_shape')}")


def _kaggle(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    _require_auth()
    exe = shutil.which("kaggle")
    if not exe:
        raise SystemExit("kaggle CLI not found; install requirements.txt")
    cmd = [exe, *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check)


def main() -> int:
    parser = argparse.ArgumentParser(description="Falcon Kaggle/GitHub orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render")
    p_render.add_argument("--slug", default=os.environ.get("KAGGLE_KERNEL_SLUG", DEFAULT_SLUG))
    p_render.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)

    sub.add_parser("validate")
    p_push = sub.add_parser("push")
    p_push.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    p_push.add_argument("--timeout", type=int, default=43200)
    sub.add_parser("status")
    sub.add_parser("logs")
    sub.add_parser("quota")
    p_output = sub.add_parser("output")
    p_output.add_argument("--dir", default="outputs")

    ns = parser.parse_args()
    slug = os.environ.get("KAGGLE_KERNEL_SLUG", DEFAULT_SLUG)

    if ns.command == "render":
        path = render(ns.slug, ns.accelerator)
        print(path)
        return 0
    if ns.command == "validate":
        validate()
        return 0
    if ns.command == "push":
        render(slug, ns.accelerator)
        validate()
        _kaggle(["kernels", "push", "-p", str(KAGGLE_DIR), "--accelerator", ns.accelerator, "--timeout", str(ns.timeout)])
        return 0
    if ns.command == "status":
        _kaggle(["kernels", "status", _kernel_ref(slug)])
        return 0
    if ns.command == "logs":
        _kaggle(["kernels", "logs", _kernel_ref(slug)])
        return 0
    if ns.command == "quota":
        _kaggle(["quota"])
        return 0
    if ns.command == "output":
        out = ROOT / ns.dir
        out.mkdir(parents=True, exist_ok=True)
        _kaggle(["kernels", "output", _kernel_ref(slug), "-p", str(out), "-o"])
        return 0
    raise AssertionError(ns.command)


if __name__ == "__main__":
    raise SystemExit(main())
