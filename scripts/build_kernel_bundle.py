#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import shutil
import zipfile


def make_embedded_package(root: Path) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((root / "flm").rglob("*.py")):
            arcname = path.relative_to(root).as_posix()
            zf.writestr(arcname, path.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def profile_env(profile: str, backend: str) -> dict[str, str]:
    if profile == "normal":
        return {
            "FLM_SMOKE": "1",
            "FLM_OUTPUT_ROOT": f"/kaggle/working/flm-v0.4-{backend}",
        }
    if profile == "fast":
        # Fast real-data accelerator validation.  The HF shards are already
        # prepared on CPU, so the Kaggle runtime only trains/evaluates.
        return {
            "FLM_SMOKE": "1",
            "FLM_OUTPUT_ROOT": f"/kaggle/working/flm-v0.4-fast-{backend}",
            "FLM_MAIN_STEPS": "2",
            "FLM_CODER_STEPS": "2",
            "FLM_COMPUTER_STEPS": "1",
            "FLM_BATCH": "2",
            "FLM_GRAD_ACCUM": "1",
            "FLM_MAIN_SEQ_LEN": "64",
            "FLM_MAIN_LAYERS": "2",
            "FLM_MAIN_HEADS": "4",
            "FLM_MAIN_EMBD": "128",
            "FLM_CODER_SEQ_LEN": "64",
            "FLM_CODER_LAYERS": "2",
            "FLM_CODER_HEADS": "4",
            "FLM_CODER_EMBD": "128",
            "FLM_EVAL_ITERS": "1",
            "FLM_COMPUTER_BATCH": "1",
            "FLM_COMPUTER_IMAGE_SIZE": "64",
            "FLM_COMPUTER_EMBD": "64",
            "FLM_COMPUTER_HEADS": "4",
            "FLM_COMPUTER_TEXT_LAYERS": "1",
            "FLM_COMPUTER_VISION_LAYERS": "1",
            "FLM_COMPUTER_TASK_LEN": "64",
            "FLM_COMPUTER_ACTION_LEN": "64",
            "FLM_COMPUTER_EVAL_BATCHES": "1",
        }
    raise ValueError(profile)


def make_wrapper(payload: str, accelerator: str, profile: str) -> str:
    chunks = "\n".join(
        f'    "{payload[i:i+100]}"'
        for i in range(0, len(payload), 100)
    )
    backend = "tpu" if accelerator.lower().startswith("tpu") else "gpu"
    env = profile_env(profile, backend)
    env_lines = "\n".join(
        f'os.environ["{key}"] = {json.dumps(value)}'
        for key, value in env.items()
    )
    return f'''# Auto-generated Falcon FLM v0.4 Kaggle bundle.
import base64
import os
from pathlib import Path
import sys

# Download/preprocessing already happened on GitHub CPU.
# This Kaggle job is accelerator-only for training + evaluation.
os.environ["FLM_ACCELERATOR"] = "{backend}"
{env_lines}

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
    p.add_argument("--profile", choices=("normal", "fast"), default="normal")
    p.add_argument("--slug", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    backend = "tpu" if args.accelerator.lower().startswith("tpu") else "gpu"
    slug = args.slug or (
        f"falcon-flm-v04-fast-{backend}"
        if args.profile == "fast"
        else "falcon-flm-v04-three-model-suite"
    )
    title = args.title or (
        f"Falcon FLM v04 Fast {backend.upper()} Trio"
        if args.profile == "fast"
        else "Falcon FLM v04 Three Model Suite"
    )

    payload = make_embedded_package(root)
    wrapper = make_wrapper(payload, args.accelerator, args.profile)
    (out / "train_suite.py").write_text(wrapper, encoding="utf-8")

    is_gpu = backend == "gpu"
    meta = {
        "id": f"{args.owner}/{slug}",
        "title": title,
        "code_file": "train_suite.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": is_gpu,
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
                "backend": backend,
                "profile": args.profile,
                "kernel": meta["id"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
