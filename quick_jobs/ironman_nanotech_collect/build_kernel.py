#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

SLUG="ironman-nanotech-render-collect"

KERNEL=r'''import json, pathlib, shutil
OUT=pathlib.Path("/kaggle/working")
IN=pathlib.Path("/kaggle/input")
OUT.mkdir(parents=True,exist_ok=True)
print("INPUT_ROOTS", [str(p) for p in IN.iterdir()] if IN.exists() else [], flush=True)

wanted=[
    "ironman_nanotech_poster.png",
    "ironman_nanotech_manifest.json",
    "kaggle_gpu_report.json",
]
found={}
for name in wanted:
    matches=list(IN.rglob(name)) if IN.exists() else []
    if matches:
        src=matches[0]
        dst=OUT/name
        shutil.copy2(src,dst)
        found[name]={"source":str(src),"size":dst.stat().st_size}
        print("COPIED",name,dst.stat().st_size,flush=True)

# Blender/FFmpeg can append frame ranges/extensions depending on output settings.
video_candidates=[]
if IN.exists():
    for p in IN.rglob("*"):
        if p.is_file() and (p.suffix.lower() in {".mp4",".mov",".mkv",".avi"} or "transform" in p.name.lower()):
            video_candidates.append(p)
print("VIDEO_CANDIDATES", [str(p) for p in video_candidates], flush=True)
if video_candidates:
    src=max(video_candidates,key=lambda p:p.stat().st_size)
    dst=OUT/"ironman_nanotech_transform.mp4"
    shutil.copy2(src,dst)
    found[dst.name]={"source":str(src),"size":dst.stat().st_size}
    print("COPIED_VIDEO",str(src),dst.stat().st_size,flush=True)

poster=OUT/"ironman_nanotech_poster.png"
if not poster.exists() or poster.stat().st_size < 1024:
    raise RuntimeError("required poster output missing")
video=OUT/"ironman_nanotech_transform.mp4"
if not video.exists() or video.stat().st_size < 1024:
    print("WARNING_NO_VIDEO_FOUND", flush=True)

(OUT/"collector_report.json").write_text(json.dumps(found,indent=2)+"\n")
print("COLLECT_DONE",json.dumps(found),flush=True)
'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--owner",required=True)
    ap.add_argument("--out",default="kernel_collect")
    ns=ap.parse_args()
    out=Path(ns.out); out.mkdir(parents=True,exist_ok=True)
    (out/"kernel.py").write_text(KERNEL,encoding="utf-8")
    meta={
        "id":f"{ns.owner}/{SLUG}",
        "title":"Ironman Nanotech Render Collect",
        "code_file":"kernel.py",
        "language":"python",
        "kernel_type":"script",
        "is_private":True,
        "enable_gpu":False,
        "enable_internet":False,
        "dataset_sources":[],
        "competition_sources":[],
        "kernel_sources":[f"{ns.owner}/ironman-nanotech-3d"],
        "model_sources":[],
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta))
if __name__=="__main__":
    main()
