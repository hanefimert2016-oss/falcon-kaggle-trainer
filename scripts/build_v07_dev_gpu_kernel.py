#!/usr/bin/env python3
from __future__ import annotations

import argparse,base64,io,json,shutil,zipfile
from pathlib import Path


def package_payload(root: Path) -> str:
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root/"flm").rglob("*.py")):
            z.writestr(p.relative_to(root).as_posix(),p.read_bytes())
    return base64.b64encode(b.getvalue()).decode("ascii")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--owner",required=True)
    ap.add_argument("--out",default="kernel_v07_dev_full")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=Path(args.out)
    if not out.is_absolute(): out=root/out
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)

    env={
        "FLM_V07_TEXT_VERSION":"v0.7-dev-text",
        "FLM_V07_COMPUTER_VERSION":"v0.7-dev-computer",
        "FLM_V07_OUTPUT_ROOT":"/kaggle/working/flm-v0.7-quality",
        "FLM_V07_MAIN_SEQ":"768",
        "FLM_V07_MAIN_LAYERS":"14","FLM_V07_MAIN_HEADS":"12","FLM_V07_MAIN_EMBD":"768",
        "FLM_V07_CODER_SEQ":"768",
        "FLM_V07_CODER_LAYERS":"12","FLM_V07_CODER_HEADS":"12","FLM_V07_CODER_EMBD":"768",
        "FLM_V07_TEXT_BATCH":"4","FLM_V07_TEXT_ACCUM":"4",
        "FLM_V07_MAIN_STEPS":"65000","FLM_V07_MAIN_SFT_STEPS":"12000",
        "FLM_V07_CODER_STEPS":"25000","FLM_V07_CODER_SFT_STEPS":"10000",
        "FLM_V07_CU_IMAGE":"224","FLM_V07_CU_TASK_LEN":"160","FLM_V07_CU_PAYLOAD_LEN":"128",
        "FLM_V07_CU_EMBD":"768","FLM_V07_CU_HEADS":"12",
        "FLM_V07_CU_TEXT_LAYERS":"4","FLM_V07_CU_VISION_LAYERS":"8",
        "FLM_V07_CU_BATCH":"4","FLM_V07_CU_STEPS":"20000",
        "FLM_V07_EVAL_BATCHES":"24","FLM_V07_CU_EVAL_EXAMPLES":"512",
        "FLM_V07_TEXT_EVAL_INTERVAL":"1000","FLM_V07_SFT_EVAL_INTERVAL":"500",
        "FLM_V07_TEXT_CHECKPOINT_INTERVAL":"4000",
    }
    p=package_payload(root)
    chunks="\n".join(f'    "{p[i:i+100]}"' for i in range(0,len(p),100))
    env_lines="\n".join(f'os.environ[{json.dumps(k)}]={json.dumps(v)}' for k,v in env.items())
    wrapper=f'''# Auto-generated FLM v0.7 DEV full GPU training.
import base64,os,sys
from pathlib import Path
{env_lines}
_PAYLOAD=(\n{chunks}\n)
z=Path("/kaggle/working/flm_v07_dev_train.zip")
z.write_bytes(base64.b64decode(_PAYLOAD))
sys.path.insert(0,str(z))
from flm.train_v07 import main
raise SystemExit(main())
'''
    (out/"train_v07_dev.py").write_text(wrapper,encoding="utf-8")
    meta={
        "id":f"{args.owner}/falcon-flm-v07-quality-gpu",
        "title":"Falcon FLM v07 Quality GPU",
        "code_file":"train_v07_dev.py",
        "language":"python","kernel_type":"script","is_private":True,
        "enable_gpu":True,"enable_internet":False,"machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[f"{args.owner}/flm-v07-dev-text",f"{args.owner}/flm-v07-dev-computer"],
        "kernel_sources":[],
        "competition_sources":[],"model_sources":[],
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"kernel":meta["id"],"payload_bytes":len(p),"steps":env}))


if __name__=="__main__":
    main()
