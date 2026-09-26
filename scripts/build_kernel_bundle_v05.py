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
    base = {
        "FLM_ACCELERATOR": "gpu",
        "FLM_V05_OUTPUT_ROOT": f"/kaggle/working/flm-v0.5-{profile}",
        "FLM_V05_MAIN_LAYERS": "14",
        "FLM_V05_MAIN_HEADS": "12",
        "FLM_V05_MAIN_EMBD": "768",
        "FLM_V05_CODER_LAYERS": "12",
        "FLM_V05_CODER_HEADS": "12",
        "FLM_V05_CODER_EMBD": "768",
        "FLM_V05_CU_EMBD": "768",
        "FLM_V05_CU_HEADS": "12",
        "FLM_V05_CU_TEXT_LAYERS": "4",
        "FLM_V05_CU_VISION_LAYERS": "8",
    }
    if profile == "pilot":
        base.update({
            "FLM_V05_MAIN_SEQ": "128",
            "FLM_V05_CODER_SEQ": "128",
            "FLM_V05_MAIN_STEPS": "3",
            "FLM_V05_CODER_STEPS": "3",
            "FLM_V05_CU_STEPS": "2",
            "FLM_V05_TEXT_BATCH": "2",
            "FLM_V05_TEXT_ACCUM": "1",
            "FLM_V05_CU_BATCH": "2",
            "FLM_V05_CU_IMAGE": "112",
            "FLM_V05_CU_TASK_LEN": "96",
            "FLM_V05_CU_ACTION_LEN": "96",
            "FLM_V05_EVAL_ITERS": "1",
            "FLM_V05_CU_EVAL_BATCHES": "1",
        })
    elif profile == "full":
        base.update({
            "FLM_V05_MAIN_SEQ": "512",
            "FLM_V05_CODER_SEQ": "512",
            "FLM_V05_MAIN_STEPS": "500",
            "FLM_V05_CODER_STEPS": "400",
            "FLM_V05_CU_STEPS": "300",
            "FLM_V05_TEXT_BATCH": "2",
            "FLM_V05_TEXT_ACCUM": "4",
            "FLM_V05_CU_BATCH": "2",
            "FLM_V05_CU_IMAGE": "224",
            "FLM_V05_CU_TASK_LEN": "192",
            "FLM_V05_CU_ACTION_LEN": "128",
        })
    else:
        raise ValueError(profile)
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--out", default="kernel_v05")
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
    wrapper = f'''# Auto-generated Falcon FLM v0.5 GPU bundle.
import base64
import os
from pathlib import Path
import sys

{env_lines}

_PAYLOAD = (
{chunks}
)
package_zip = Path("/kaggle/working/flm_v05_package.zip")
package_zip.write_bytes(base64.b64decode(_PAYLOAD))
sys.path.insert(0, str(package_zip))
from flm.train_v05 import main
raise SystemExit(main())
'''
    (out / "train_v05.py").write_text(wrapper, encoding="utf-8")

    slug = f"falcon-flm-v05-{args.profile}-gpu"
    title = f"Falcon FLM v05 {args.profile.title()} GPU"
    meta = {
        "id": f"{args.owner}/{slug}",
        "title": title,
        "code_file": "train_v05.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{args.owner}/flm-hf-v05"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kernel": meta["id"], "profile": args.profile, "payload_bytes": len(payload)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
