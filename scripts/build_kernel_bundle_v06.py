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
        for p in sorted((root/"flm").rglob("*.py")):
            zf.writestr(p.relative_to(root).as_posix(), p.read_bytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def env_for(profile: str) -> dict[str, str]:
    base = {
        "FLM_ACCELERATOR": "gpu",
        "FLM_V06_OUTPUT_ROOT": f"/kaggle/working/flm-v0.6-{profile}",
        "FLM_V06_SEQ": "512",
        "FLM_V06_TEXT_BATCH": "4",
        "FLM_V06_TEXT_ACCUM": "2",
        "FLM_V06_MAIN_EPOCHS": "2",
        "FLM_V06_SFT_EPOCHS": "1",
        "FLM_V06_CODER_EPOCHS": "1",
        "FLM_V06_CU_EPOCHS": "2",
        "FLM_V06_CU_BATCH": "2",
        "FLM_V06_EVAL_ITERS": "8",
        "FLM_V05_CU_EVAL_BATCHES": "16",
    }
    if profile == "pilot":
        base.update({
            "FLM_V06_MAX_PRETRAIN_STEPS": "3",
            "FLM_V06_MAX_SFT_STEPS": "2",
            "FLM_V06_MAX_CU_STEPS": "3",
            "FLM_V06_EVAL_ITERS": "1",
            "FLM_V05_CU_EVAL_BATCHES": "1",
        })
    elif profile == "full":
        base.update({
            "FLM_V06_MAX_PRETRAIN_STEPS": "0",
            "FLM_V06_MAX_SFT_STEPS": "0",
            "FLM_V06_MAX_CU_STEPS": "0",
        })
    else:
        raise ValueError(profile)
    return base


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--owner",required=True)
    ap.add_argument("--profile",choices=("pilot","full"),default="pilot")
    ap.add_argument("--out",default="kernel_v06")
    args=ap.parse_args()

    root=Path(__file__).resolve().parents[1]
    out=Path(args.out)
    if not out.is_absolute():
        out=root/out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    payload=package_payload(root)
    chunks="\n".join(f'    "{payload[i:i+100]}"' for i in range(0,len(payload),100))
    env_lines="\n".join(f'os.environ["{k}"] = {json.dumps(v)}' for k,v in env_for(args.profile).items())
    wrapper=f'''# Auto-generated Falcon FLM v0.6 bundle.
import base64
import os
from pathlib import Path
import sys

{env_lines}

_PAYLOAD = (
{chunks}
)
package_zip=Path("/kaggle/working/flm_v06_package.zip")
package_zip.write_bytes(base64.b64decode(_PAYLOAD))
sys.path.insert(0,str(package_zip))
from flm.train_v06 import main
raise SystemExit(main())
'''
    (out/"train_v06.py").write_text(wrapper,encoding="utf-8")

    slug=f"falcon-flm-v06-{args.profile}-gpu"
    meta={
        "id":f"{args.owner}/{slug}",
        "title":f"Falcon FLM v06 {args.profile.title()} GPU",
        "code_file":"train_v06.py",
        "language":"python",
        "kernel_type":"script",
        "is_private":True,
        "enable_gpu":True,
        "enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[f"{args.owner}/flm-hf-v06",f"{args.owner}/flm-hf-v05"],
        "competition_sources":[],
        "kernel_sources":[],
        "model_sources":[],
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"kernel":meta["id"],"profile":args.profile,"payload_bytes":len(payload)}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
