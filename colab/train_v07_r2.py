#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(*args:str)->None:
    print("+"," ".join(args),flush=True)
    subprocess.run(list(args),check=True)


def valid_revision(path:Path)->bool:
    meta=path/"sources.json"
    if not meta.is_file():
        return False
    try:
        obj=json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return obj.get("pipeline_version")=="v0.7-dev-text" and int(obj.get("data_revision",0))>=7


def ensure_text_dataset(ref:str,dest:Path)->None:
    if valid_revision(dest):
        print("dataset_ready",ref,dest,flush=True)
        return
    if not (os.environ.get("KAGGLE_API_TOKEN") or Path.home().joinpath(".kaggle/kaggle.json").is_file()):
        raise SystemExit("KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json is required to download FLM r4 text data.")
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    dest.mkdir(parents=True,exist_ok=True)
    run(sys.executable,"-m","kaggle","datasets","download","-d",ref,"-p",str(dest),"--unzip")
    if not valid_revision(dest):
        raise RuntimeError(f"downloaded dataset is not FLM semantic r4: {ref}")


def gpu_profile():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Colab GPU is not enabled.")
    props=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    min_gib=min(int(p.total_memory//1024**3) for p in props)
    print("GPUs:",[(p.name,round(p.total_memory/1024**3,2)) for p in props],flush=True)
    if min_gib>=40:
        return 2,8
    if min_gib>=14:
        return 1,16
    return 1,32


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--drive-root",default="/content/drive/MyDrive/FalconFLM")
    ap.add_argument("--data-root",default="/content/flm-v07-r4-data")
    ap.add_argument("--owner",default=os.environ.get("KAGGLE_OWNER","mertsigma"))
    ap.add_argument("--skip-drive-mount",action="store_true")
    args=ap.parse_args()

    if not args.skip_drive_mount:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as exc:
            if not Path("/content/drive/MyDrive").exists():
                raise SystemExit(f"Google Drive mount failed: {exc}")

    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0,str(root))
    run(sys.executable,"-m","pip","install","-q","--upgrade",
        "kaggle>=2.2.1","tokenizers>=0.20","numpy>=1.26","pillow>=10")

    data_root=Path(args.data_root)
    text_dir=data_root/"text"
    ensure_text_dataset(f"{args.owner}/flm-v07-semantic-text-r7",text_dir)

    batch,accum=gpu_profile()
    out=Path(args.drive_root)/"runs"/"flm-v0.7-semantic"
    out.mkdir(parents=True,exist_ok=True)

    env=os.environ.copy()
    env.update({
        "PYTHONPATH":str(root),
        "FLM_ACCELERATOR":"gpu",
        "FLM_V07_DATA_ROOT":str(text_dir),
        "FLM_V07_TEXT_VERSION":"v0.7-dev-text",
        "FLM_INTERFACE_OUTPUT_ROOT":str(out),
        "FLM_V07_RESUME":"1",
        "FLM_INTERFACE_SEQ":"4096",
        "FLM_INTERFACE_LAYERS":"16",
        "FLM_INTERFACE_HEADS":"12",
        "FLM_INTERFACE_KV_HEADS":"4",
        "FLM_INTERFACE_EMBD":"768",
        "FLM_INTERFACE_BATCH":str(batch),
        "FLM_INTERFACE_ACCUM":str(accum),
        "FLM_INTERFACE_GENERAL_SEQ":"1024",
        "FLM_INTERFACE_CODE_SEQ":"2048",
        "FLM_INTERFACE_SFT_SEQ":"4096",
        "FLM_INTERFACE_GENERAL_EPOCHS":"1.0",
        "FLM_INTERFACE_CODE_EPOCHS":"1.0",
        "FLM_INTERFACE_SFT_EPOCHS":"1.0",
        "FLM_INTERFACE_GRADIENT_CHECKPOINTING":"1",
        "FLM_INTERFACE_CKPT_INTERVAL":"250",
        "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True",
    })
    print(json.dumps({
        "architecture":"one InterfaceTransformer + training-free FLM Core",
        "data":str(text_dir),
        "output":str(out),
        "batch":batch,
        "accum":accum,
    },indent=2),flush=True)
    subprocess.run(
        [sys.executable,"-u","-m","flm.train_interface_v07"],
        check=True,env=env,
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
