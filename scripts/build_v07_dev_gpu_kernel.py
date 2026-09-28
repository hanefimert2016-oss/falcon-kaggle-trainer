#!/usr/bin/env python3
from __future__ import annotations

import argparse,base64,io,json,shutil,zipfile
from pathlib import Path


def package_payload(root:Path)->str:
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root/"flm").rglob("*.py")):
            z.writestr(p.relative_to(root).as_posix(),p.read_bytes())
    return base64.b64encode(b.getvalue()).decode("ascii")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--owner",required=True)
    ap.add_argument("--out",default="kernel_v07_semantic_interface")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=Path(args.out)
    if not out.is_absolute():
        out=root/out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    env={
        "FLM_V07_TEXT_VERSION":"v0.7-dev-text",
        "FLM_INTERFACE_OUTPUT_ROOT":"/kaggle/working/flm-v0.7-semantic",
        "FLM_ACCELERATOR":"gpu",
        "FLM_INTERFACE_SEQ":"4096",
        "FLM_INTERFACE_LAYERS":"16",
        "FLM_INTERFACE_HEADS":"12",
        "FLM_INTERFACE_KV_HEADS":"4",
        "FLM_INTERFACE_EMBD":"768",
        "FLM_INTERFACE_BATCH":"2",
        "FLM_INTERFACE_ACCUM":"8",
        "FLM_INTERFACE_GENERAL_SEQ":"1024",
        "FLM_INTERFACE_CODE_SEQ":"2048",
        "FLM_INTERFACE_SFT_SEQ":"4096",
        "FLM_INTERFACE_GENERAL_EPOCHS":"1.0",
        "FLM_INTERFACE_CODE_EPOCHS":"1.0",
        "FLM_INTERFACE_SFT_EPOCHS":"1.0",
        "FLM_INTERFACE_GRADIENT_CHECKPOINTING":"1",
        "FLM_INTERFACE_CKPT_INTERVAL":"500",
        "FLM_INTERFACE_EVAL_INTERVAL":"500",
        "FLM_INTERFACE_SFT_EVAL_INTERVAL":"250",
        "FLM_INTERFACE_EVAL_BATCHES":"16",
        "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True",
    }
    payload=package_payload(root)
    chunks="\n".join(f'    "{payload[i:i+100]}"' for i in range(0,len(payload),100))
    env_lines="\n".join(f'os.environ[{json.dumps(k)}]={json.dumps(v)}' for k,v in env.items())
    wrapper=f'''# Auto-generated Semantic FLM v0.7 SINGLE Transformer training.
import base64,os,sys
from pathlib import Path
{env_lines}
_PAYLOAD=(\n{chunks}\n)
z=Path("/kaggle/working/flm_semantic_interface.zip")
z.write_bytes(base64.b64decode(_PAYLOAD))
sys.path.insert(0,str(z))
from flm.train_interface_v07 import main
raise SystemExit(main())
'''
    (out/"train_interface_v07.py").write_text(wrapper,encoding="utf-8")
    meta={
        "id":f"{args.owner}/falcon-flm-v07-semantic-interface-gpu",
        "title":"Falcon FLM v07 Semantic Interface Transformer",
        "code_file":"train_interface_v07.py",
        "language":"python",
        "kernel_type":"script",
        "is_private":True,
        "enable_gpu":True,
        "enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[f"{args.owner}/flm-v07-semantic-text-r3"],
        "kernel_sources":[],
        "competition_sources":[],
        "model_sources":[],
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({
        "kernel":meta["id"],
        "transformer_count":1,
        "dataset_sources":meta["dataset_sources"],
        "payload_bytes":len(payload),
        "env":env,
    }))


if __name__=="__main__":
    main()
