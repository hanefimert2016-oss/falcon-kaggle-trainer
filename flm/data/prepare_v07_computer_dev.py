#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil

from datasets import load_dataset

from flm.data.prepare_v07 import ImageShardWriter, prepare_rexx
from flm.data.prepare_v05 import (
    ZipShardWriter,
    add_groundcua,
    add_salesforce,
)


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def payload_from_action(raw: str, op: str) -> str:
    try:
        tree=ast.parse(str(raw or ""))
    except Exception:
        return ""
    for call in [n for n in ast.walk(tree) if isinstance(n,ast.Call)]:
        f=call.func
        name=f.attr if isinstance(f,ast.Attribute) else (f.id if isinstance(f,ast.Name) else "")
        if op=="TYPE" and name in {"write","typewrite"} and call.args:
            v=_literal(call.args[0])
            return str(v) if isinstance(v,str) else ""
        if op=="KEY" and name in {"hotkey","press"}:
            vals=[_literal(x) for x in call.args]
            return "+".join(str(x) for x in vals if isinstance(x,(str,int)))
        if op=="SCROLL" and name in {"scroll","hscroll"} and call.args:
            v=_literal(call.args[0])
            if isinstance(v,(int,float)):
                return f"{int(v)},0" if name=="hscroll" else f"0,{int(v)}"
    return ""


def append_legacy_grounding(out: Path, manifest: Path, ground_target: int, salesforce_target: int):
    rows=[]
    writer=ZipShardWriter(out,max_images=750)
    stats={}
    try:
        stats["groundcua"]=add_groundcua(rows,writer,ground_target)
        print("V07_DEV_CU groundcua",json.dumps(stats["groundcua"]),flush=True)
        stats["salesforce"]=add_salesforce(rows,writer,salesforce_target)
        print("V07_DEV_CU salesforce",json.dumps(stats["salesforce"]),flush=True)
    finally:
        writer.close()

    accepted=0
    ops={}
    with manifest.open("a",encoding="utf-8") as fh:
        for i,r in enumerate(rows):
            op=str(r.get("operation") or "").upper()
            if op not in {"MOVE","CLICK","DOUBLE_CLICK","RIGHT_CLICK","SCROLL","TYPE","KEY","KEY_DOWN","KEY_UP"}:
                continue
            coord=r.get("coord") if r.get("coord_valid") else None
            if op in {"MOVE","CLICK","DOUBLE_CLICK","RIGHT_CLICK"} and coord is None:
                continue
            payload=payload_from_action(r.get("action",""),op)
            if op in {"TYPE","KEY","KEY_DOWN","KEY_UP","SCROLL"} and not payload:
                continue
            rec={
                "archive":r["archive"],
                "image":r["image"],
                "task":r.get("task",""),
                "operation":op,
                "coord":coord,
                "coord2":None,
                "payload":payload,
                "domain":r.get("domain","legacy:desktop"),
                "source":r.get("source","legacy"),
                "episode_id":f"ground:{r.get('source','legacy')}:{i}",
                "step":0,
            }
            fh.write(json.dumps(rec,ensure_ascii=False)+"\n")
            accepted+=1
            ops[op]=ops.get(op,0)+1
    stats["accepted"]=accepted
    stats["ops"]=ops
    stats["image_bytes"]=writer.total_bytes
    return stats


def add_showui(out: Path, manifest: Path, limit: int):
    ds=load_dataset("showlab/ShowUI-desktop",split="train",streaming=True)
    writer=ImageShardWriter(out,prefix="showui_images",max_images=600)
    added=0
    types={}
    with manifest.open("a",encoding="utf-8") as fh:
        try:
            for row in ds:
                image=row.get("image")
                point=row.get("point")
                bbox=row.get("bbox")
                instruction=str(row.get("instruction") or "").strip()
                if image is None or not point or len(point)<2 or not instruction:
                    continue
                x,y=float(point[0]),float(point[1])
                if x>1.5 or y>1.5:
                    w,h=image.size
                    x/=max(1,w); y/=max(1,h)
                x=min(1.0,max(0.0,x)); y=min(1.0,max(0.0,y))
                typ=str(row.get("type") or "click")
                archive,member,_,_=writer.add(image,f"showui_{added:07d}.jpg")
                rec={
                    "archive":archive,"image":member,"task":instruction,
                    "operation":"CLICK","coord":[x,y],"coord2":None,"payload":"",
                    "domain":"showui:"+typ,"source":"showlab/ShowUI-desktop",
                    "episode_id":f"showui:{added}","step":0,
                }
                fh.write(json.dumps(rec,ensure_ascii=False)+"\n")
                added+=1
                types[typ]=types.get(typ,0)+1
                if added>=limit:
                    break
        finally:
            writer.close()
    if added < min(5_000,int(limit*.80)):
        raise RuntimeError(f"ShowUI underfilled {added}/{limit}")
    return {"examples":added,"types":types}


def add_click100k(out: Path, manifest: Path, limit: int):
    ds=load_dataset("mlfoundations/Click-100k",split="train",streaming=True)
    writer=ImageShardWriter(out,prefix="click100k_images",max_images=600)
    added=0
    with manifest.open("a",encoding="utf-8") as fh:
        try:
            for row in ds:
                images=row.get("images")
                image=(images[0] if isinstance(images,list) and images else None)
                box=row.get("normalized_bbox")
                prompt=str(row.get("easyr1_prompt") or "").strip()
                if image is None or not box or len(box)!=4 or not prompt:
                    continue
                x=(float(box[0])+float(box[2]))/2
                y=(float(box[1])+float(box[3]))/2
                archive,member,_,_=writer.add(image,f"click100k_{added:07d}.jpg")
                # Keep the final task phrase and remove most benchmark boilerplate.
                task=prompt.split("<image>")[-1].strip() or prompt[-900:]
                rec={
                    "archive":archive,"image":member,"task":task[:1800],
                    "operation":"CLICK","coord":[min(1,max(0,x)),min(1,max(0,y))],
                    "coord2":None,"payload":"","domain":"click100k:desktop",
                    "source":"mlfoundations/Click-100k",
                    "episode_id":f"click100k:{added}","step":0,
                }
                fh.write(json.dumps(rec,ensure_ascii=False)+"\n")
                added+=1
                if added>=limit:
                    break
        finally:
            writer.close()
    if added < int(limit*.80):
        raise RuntimeError(f"Click-100k underfilled {added}/{limit}")
    return {"examples":added}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="/kaggle/working/flm-v0.7-dev-computer")
    ap.add_argument("--rexx-trajectories",type=int,default=5000)
    ap.add_argument("--groundcua",type=int,default=18000)
    ap.add_argument("--salesforce",type=int,default=4000)
    ap.add_argument("--showui",type=int,default=7496)
    ap.add_argument("--click100k",type=int,default=15000)
    args=ap.parse_args()

    out=Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stats={}
    stats["rexx"]=prepare_rexx(out,args.rexx_trajectories)
    manifest=out/"cu07_manifest.jsonl"
    stats["legacy_grounding"]=append_legacy_grounding(out,manifest,args.groundcua,args.salesforce)
    stats["showui"]=add_showui(out,manifest,args.showui)
    stats["click100k"]=add_click100k(out,manifest,args.click100k)

    rows=0
    ops={}
    sources={}
    domains=set()
    with manifest.open("r",encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            r=json.loads(line); rows+=1
            op=r["operation"]; ops[op]=ops.get(op,0)+1
            src=r["source"]; sources[src]=sources.get(src,0)+1
            domains.add(r.get("domain",""))
    if rows < 30000:
        raise RuntimeError(f"v0.7 dev computer dataset too small: {rows}")
    if ops.get("CLICK",0)<20000 or ops.get("KEY",0)<100 or ops.get("TYPE",0)<10:
        raise RuntimeError(f"insufficient action diversity: {ops}")

    summary={
        "pipeline_version":"v0.7-dev-computer",
        "training_pipeline":"v0.7",
        "stats":{
            "examples":rows,
            "ops":ops,
            "sources":sources,
            "domains":len(domains),
            "parts":stats,
        },
    }
    (out/"sources.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("V07_DEV_COMPUTER_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
