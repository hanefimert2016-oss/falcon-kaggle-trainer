#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True)
    p.add_argument("--accelerator", default="NvidiaTeslaT4")
    p.add_argument("--out", default="kernel_bundle")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copytree(root / "flm", out / "flm")
    (out / "train_suite.py").write_text(
        "from flm.train_suite import main\nraise SystemExit(main())\n",
        encoding="utf-8",
    )

    meta = {
        "id": f"{args.owner}/falcon-flm-v04-three-model-suite",
        "title": "Falcon FLM v04 Three Model Suite",
        "code_file": "train_suite.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": args.accelerator.startswith("Nvidia"),
        "enable_internet": False,
        "machine_shape": args.accelerator,
        "dataset_sources": [f"{args.owner}/flm-hf-v04"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
