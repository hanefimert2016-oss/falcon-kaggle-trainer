#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import shutil
import zipfile


def package_payload(root: Path) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted((root / "flm").rglob("*.py")):
            zf.writestr(p.relative_to(root).as_posix(), p.read_bytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def env_for(profile: str) -> dict[str, str]:
    common = {
        "FLM_V07_MAIN_SEQ": "512",
        "FLM_V07_MAIN_LAYERS": "14",
        "FLM_V07_MAIN_HEADS": "12",
        "FLM_V07_MAIN_EMBD": "768",
        "FLM_V07_CODER_SEQ": "512",
        "FLM_V07_CODER_LAYERS": "12",
        "FLM_V07_CODER_HEADS": "12",
        "FLM_V07_CODER_EMBD": "768",
        "FLM_V07_CU_IMAGE": "224",
        "FLM_V07_CU_TASK_LEN": "160",
        "FLM_V07_CU_PAYLOAD_LEN": "128",
        "FLM_V07_CU_EMBD": "768",
        "FLM_V07_CU_HEADS": "12",
        "FLM_V07_CU_TEXT_LAYERS": "4",
        "FLM_V07_CU_VISION_LAYERS": "8",
        "FLM_V07_EVAL_BATCHES": "4",
        "FLM_V07_CU_EVAL_EXAMPLES": "64",
    }
    if profile == "pilot":
        common.update({
            # Same production shapes/batches; only the number of optimizer
            # steps is shortened so memory regressions are caught before full.
            "FLM_V07_OUTPUT_ROOT": "/kaggle/working/flm-v0.7-pilot",
            "FLM_V07_TEXT_BATCH": "8",
            "FLM_V07_TEXT_ACCUM": "2",
            "FLM_V07_MAIN_STEPS": "3",
            "FLM_V07_MAIN_SFT_STEPS": "3",
            "FLM_V07_CODER_STEPS": "3",
            "FLM_V07_CODER_SFT_STEPS": "3",
            "FLM_V07_CU_BATCH": "4",
            "FLM_V07_CU_STEPS": "3",
        })
    elif profile == "full":
        common.update({
            "FLM_V07_OUTPUT_ROOT": "/kaggle/working/flm-v0.7-full",
            "FLM_V07_TEXT_BATCH": "8",
            "FLM_V07_TEXT_ACCUM": "2",
            "FLM_V07_MAIN_STEPS": "25000",
            "FLM_V07_MAIN_SFT_STEPS": "3500",
            "FLM_V07_CODER_STEPS": "15000",
            "FLM_V07_CODER_SFT_STEPS": "3500",
            "FLM_V07_CU_BATCH": "4",
            "FLM_V07_CU_STEPS": "8000",
            "FLM_V07_EVAL_BATCHES": "8",
            "FLM_V07_CU_EVAL_EXAMPLES": "128",
        })
    else:
        raise ValueError(profile)
    return common


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--out", default="kernel_v07")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    payload = package_payload(root)
    chunks = "\n".join(f'    "{payload[i:i+100]}"' for i in range(0, len(payload), 100))
    env_lines = "\n".join(
        f'os.environ["{k}"] = {json.dumps(v)}'
        for k, v in env_for(args.profile).items()
    )
    wrapper = f'''# Auto-generated FLM v0.7 GPU bundle.
import base64
import os
from pathlib import Path
import sys

{env_lines}

_PAYLOAD = (
{chunks}
)
package_zip = Path("/kaggle/working/flm_v07_package.zip")
package_zip.write_bytes(base64.b64decode(_PAYLOAD))
sys.path.insert(0, str(package_zip))
from flm.train_v07 import main
raise SystemExit(main())
'''
    code = out / "train_v07.py"
    code.write_text(wrapper, encoding="utf-8")

    slug = f"falcon-flm-v07-{args.profile}-gpu"
    meta = {
        "id": f"{args.owner}/{slug}",
        "title": f"Falcon FLM v07 {args.profile.title()} GPU",
        "code_file": code.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            f"{args.owner}/flm-hf-v07",
            f"{args.owner}/flm-hf-v05",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kernel": meta["id"], "profile": args.profile, "payload_bytes": len(payload)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
