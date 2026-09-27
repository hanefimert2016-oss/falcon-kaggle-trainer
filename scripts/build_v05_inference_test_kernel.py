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


TEST_CODE = r'''
import base64, io, json, math, os, random, sys, zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from flm.models.text_lm import ByteCausalLM, TextConfig
from flm.models.computer_use_v2 import ComputerUseV2, ComputerUseV2Config

OPS = ["CLICK","TYPE","KEY","SCROLL","MOVE","DRAG","GAME_ACTION","OTHER"]


def find_full_root() -> Path:
    roots=[]
    for p in Path("/kaggle/input").rglob("suite_metrics.json"):
        try:
            x=json.loads(p.read_text())
        except Exception:
            continue
        if x.get("pipeline_version")=="v0.5" and x.get("accelerator")=="gpu":
            if (p.parent/"main/checkpoint.pt").is_file() and (p.parent/"computer_use/checkpoint.pt").is_file() and (p.parent/"coder/checkpoint.pt").is_file():
                roots.append(p.parent)
    if not roots:
        raise RuntimeError("final v0.5 checkpoint root not found under /kaggle/input")
    roots.sort(key=lambda p: ("full" not in str(p).lower(), len(str(p))))
    root=roots[0]
    print("CHECKPOINT_ROOT",root,flush=True)
    return root


def find_data_root() -> Path:
    for p in Path("/kaggle/input").rglob("sources.json"):
        try:
            x=json.loads(p.read_text())
        except Exception:
            continue
        if x.get("pipeline_version")=="v0.5" and (p.parent/"computer_manifest.jsonl").is_file():
            print("DATA_ROOT",p.parent,flush=True)
            return p.parent
    raise RuntimeError("v0.5 data root not found")


def load_text_model(path: Path):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    cfg=TextConfig(**ck["config"])
    model=ByteCausalLM(cfg)
    missing,unexpected=model.load_state_dict(ck["model"],strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch missing={missing} unexpected={unexpected}")
    return model,cfg,ck


@torch.inference_mode()
def generate_text(model,cfg,prompt,max_new=160,temperature=0.8,top_k=40):
    device=next(model.parameters()).device
    ids=list(prompt.encode("utf-8","ignore"))
    for _ in range(max_new):
        ctx=ids[-cfg.seq_len:]
        x=torch.tensor([ctx],dtype=torch.long,device=device)
        logits,_=model(x)
        z=logits[0,-1].float()/max(temperature,1e-5)
        k=min(top_k,z.numel())
        vals,idx=torch.topk(z,k)
        probs=torch.softmax(vals,dim=-1)
        pick=idx[torch.multinomial(probs,1)].item()
        ids.append(int(pick))
    return bytes(ids).decode("utf-8","replace")


@torch.inference_mode()
def eval_bytes(model,cfg,path: Path,samples=8):
    raw=path.read_bytes()
    device=next(model.parameters()).device
    rng=random.Random(20260927)
    losses=[]
    for _ in range(samples):
        start=rng.randrange(0,max(1,len(raw)-cfg.seq_len-2))
        chunk=raw[start:start+cfg.seq_len+1]
        if len(chunk)<cfg.seq_len+1:
            continue
        x=torch.tensor([list(chunk[:-1])],dtype=torch.long,device=device)
        y=torch.tensor([list(chunk[1:])],dtype=torch.long,device=device)
        _,loss=model(x,y)
        losses.append(float(loss))
    loss=sum(losses)/len(losses)
    return {"loss":loss,"perplexity":math.exp(min(loss,20)),"samples":len(losses)}


class ImageStore:
    def __init__(self,root):
        self.root=Path(root)
        self.zips={}
    def get(self,archive,member,size):
        zp=self.root/archive
        if zp.is_file():
            z=self.zips.setdefault(archive,zipfile.ZipFile(zp))
            raw=z.read(member)
            im=Image.open(io.BytesIO(raw))
        else:
            d=self.root/Path(archive).stem
            p=d/member
            if not p.is_file():
                matches=list(d.rglob(member))
                if not matches:
                    raise FileNotFoundError((archive,member))
                p=matches[0]
            im=Image.open(p)
        with im:
            arr=np.asarray(im.convert("RGB").resize((size,size)),dtype=np.float32)/255.0
        arr=(arr-0.5)/0.5
        return torch.from_numpy(arr).permute(2,0,1).contiguous()


def task_bytes(text,cfg):
    b=list(str(text).encode("utf-8","ignore")[:cfg.task_len])
    b += [cfg.pad_token]*(cfg.task_len-len(b))
    return torch.tensor([b],dtype=torch.long)


def action_seed(cfg):
    x=[cfg.bos_token]+[cfg.pad_token]*(cfg.action_len-1)
    return torch.tensor([x],dtype=torch.long)


def load_cu(path):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    cfg=ComputerUseV2Config(**ck["config"])
    model=ComputerUseV2(cfg)
    missing,unexpected=model.load_state_dict(ck["model"],strict=False)
    if missing or unexpected:
        raise RuntimeError(f"CU state mismatch missing={missing} unexpected={unexpected}")
    return model,cfg,ck


@torch.inference_mode()
def test_cu(model,cfg,ck,data_root,limit=32):
    device=next(model.parameters()).device
    rows=[json.loads(x) for x in (data_root/"computer_manifest.jsonl").read_text().splitlines() if x.strip()]
    rng=random.Random(12345)
    rng.shuffle(rows)
    domains=ck["domains"]
    id_to_domain={v:k for k,v in domains.items()}
    store=ImageStore(data_root)
    tested=op_valid=op_correct=coord_valid=0
    coord_errs=[]
    examples=[]
    for r in rows:
        try:
            image=store.get(r["archive"],r["image"],cfg.image_size).unsqueeze(0).to(device)
        except Exception:
            continue
        task=task_bytes(r.get("task",""),cfg).to(device)
        ain=action_seed(cfg).to(device)
        out,_,_=model(image,task,ain)
        pred_op=int(out["op_logits"].argmax(-1).item())
        pred_coord=out["coord"][0].float().cpu().tolist()
        pred_domain=int(out["domain_logits"].argmax(-1).item())
        truth_op=str(r.get("operation","OTHER"))
        if r.get("op_valid",False):
            op_valid+=1
            op_correct += int(OPS[pred_op]==truth_op)
        if r.get("coord_valid",False) and r.get("coord"):
            coord_valid+=1
            t=r["coord"]
            coord_errs.append(((pred_coord[0]-t[0])**2+(pred_coord[1]-t[1])**2)**0.5)
        if len(examples)<8:
            examples.append({
                "task":r.get("task","")[:180],
                "source":r.get("source"),
                "truth_op":truth_op,
                "pred_op":OPS[pred_op],
                "truth_coord":r.get("coord"),
                "pred_coord":[round(x,4) for x in pred_coord],
                "truth_domain":r.get("domain"),
                "pred_domain":id_to_domain.get(pred_domain,f"id:{pred_domain}"),
            })
        tested+=1
        if tested>=limit:
            break
    if tested<8:
        raise RuntimeError(f"too few CU samples tested: {tested}")
    return {
        "tested":tested,
        "op_valid":op_valid,
        "op_accuracy":op_correct/max(1,op_valid),
        "coord_valid":coord_valid,
        "coord_mean_l2":sum(coord_errs)/max(1,len(coord_errs)),
        "examples":examples,
    }


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device=torch.device("cuda:0")
    print("CUDA",torch.cuda.get_device_name(0),flush=True)

    root=find_full_root()
    data=find_data_root()
    results={"gpu":torch.cuda.get_device_name(0)}

    main_model,main_cfg,main_ck=load_text_model(root/"main/checkpoint.pt")
    main_model.to(device).eval()
    results["main"]={
        "params":sum(p.numel() for p in main_model.parameters()),
        "eval":eval_bytes(main_model,main_cfg,data/"main_train.bin",8),
        "generations":[
            generate_text(main_model,main_cfg,"The future of artificial intelligence is ",160),
            generate_text(main_model,main_cfg,"A computer operating system is ",160),
        ],
    }
    print("MAIN_RESULT",json.dumps(results["main"],ensure_ascii=False),flush=True)
    del main_model
    torch.cuda.empty_cache()

    coder,coder_cfg,coder_ck=load_text_model(root/"coder/checkpoint.pt")
    coder.to(device).eval()
    results["coder"]={
        "params":sum(p.numel() for p in coder.parameters()),
        "eval":eval_bytes(coder,coder_cfg,data/"coder_train.bin",8),
        "generations":[
            generate_text(coder,coder_cfg,"<instruction>Write a Python function that adds two integers.</instruction>\n<code>\n",220),
            generate_text(coder,coder_cfg,"<instruction>Create a Python class for a simple stack.</instruction>\n<code>\n",220),
        ],
    }
    print("CODER_RESULT",json.dumps(results["coder"],ensure_ascii=False),flush=True)
    del coder
    torch.cuda.empty_cache()

    cu,cu_cfg,cu_ck=load_cu(root/"computer_use/checkpoint.pt")
    cu.to(device).eval()
    results["computer_use"]={"params":sum(p.numel() for p in cu.parameters()),**test_cu(cu,cu_cfg,cu_ck,data,32)}
    print("CU_RESULT",json.dumps(results["computer_use"],ensure_ascii=False),flush=True)

    out=Path("/kaggle/working/flm-v05-inference-test")
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print("INFERENCE_TEST_COMPLETE",out,flush=True)


if __name__=="__main__":
    main()
'''


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--owner",required=True)
    ap.add_argument("--out",default="kernel_v05_inference")
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
    wrapper=f'''# Auto-generated FLM v0.5 final inference test.
import base64,sys
from pathlib import Path
_PAYLOAD=(\n{chunks}\n)
package_zip=Path("/kaggle/working/flm_v05_test_package.zip")
package_zip.write_bytes(base64.b64decode(_PAYLOAD))
sys.path.insert(0,str(package_zip))
{TEST_CODE}
'''
    (out/"test_v05.py").write_text(wrapper,encoding="utf-8")
    meta={
        "id":f"{args.owner}/falcon-flm-v05-final-inference-test",
        "title":"Falcon FLM v05 Final Inference Test",
        "code_file":"test_v05.py",
        "language":"python",
        "kernel_type":"script",
        "is_private":True,
        "enable_gpu":True,
        "enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[f"{args.owner}/flm-hf-v05"],
        "competition_sources":[],
        "kernel_sources":[f"{args.owner}/falcon-flm-v05-full-gpu"],
        "model_sources":[],
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"kernel":meta["id"],"payload_bytes":len(payload),"out":str(out)}))

if __name__=="__main__":
    main()
