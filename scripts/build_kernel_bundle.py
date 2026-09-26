#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import shutil
import textwrap
import zipfile


def make_embedded_package(root: Path) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((root / "flm").rglob("*.py")):
            arcname = path.relative_to(root).as_posix()
            zf.writestr(arcname, path.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_wrapper(payload: str) -> str:
    chunks = "\n".join(
        f'    "{payload[i:i+100]}"'
        for i in range(0, len(payload), 100)
    )
    return f'''# Auto-generated Falcon FLM v0.4 Kaggle bundle.
import base64
import os
from pathlib import Path
import sys

_PAYLOAD = (
{chunks}
)

package_zip = Path("/kaggle/working/flm_v04_package.zip")
package_zip.write_bytes(base64.b64decode(_PAYLOAD))
sys.path.insert(0, str(package_zip))

from flm.train_suite import main

raise SystemExit(main())
'''


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True)
    p.add_argument("--accelerator", default="NvidiaTeslaT4")
    p.add_argument("--out", default="kernel_bundle")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    payload = make_embedded_package(root)
    wrapper = make_wrapper(payload)
    (out / "train_suite.py").write_text(wrapper, encoding="utf-8")

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
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "bundle": str(out),
                "embedded_package_base64_bytes": len(payload),
                "code_file_bytes": (out / "train_suite.py").stat().st_size,
                "accelerator": args.accelerator,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
