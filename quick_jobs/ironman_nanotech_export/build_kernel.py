#!/usr/bin/env python3
# trigger: 2026-10-03T20:50Z
from __future__ import annotations
import argparse, base64, json
from pathlib import Path

BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz"
SLUG="ironman-nanotech-export"

TEMPLATE=r'''import base64, json, os, pathlib, shutil, subprocess, tarfile, urllib.request
OUT=pathlib.Path("/kaggle/working"); OUT.mkdir(parents=True,exist_ok=True)
subprocess.run(["nvidia-smi"],check=False)
url=__URL__
archive=OUT/"blender.tar.xz"; bd=OUT/"blender"
if not bd.exists():
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36"})
    with urllib.request.urlopen(req,timeout=120) as src, archive.open("wb") as dst:
        shutil.copyfileobj(src,dst,length=1024*1024)
    with tarfile.open(archive,"r:xz") as tf: tf.extractall(OUT)
    dirs=sorted(OUT.glob("blender-4.5.*-linux-x64"))
    if not dirs: raise RuntimeError("Blender directory not found")
    dirs[0].rename(bd)
blender=bd/"blender"; blender.chmod(0o755)
scene=OUT/"scene.py"; scene.write_bytes(base64.b64decode(__SCENE__))
env=os.environ.copy(); env["IRONMAN_OUT"]=str(OUT)
subprocess.check_call([str(blender),"-b","--python",str(scene)],env=env)
expected=["ironman_nanotech_scene.blend","ironman_nanotech_animated.glb","ironman_nanotech_manifest.json"]
sizes={}
for name in expected:
    p=OUT/name
    if not p.exists() or p.stat().st_size<1024: raise RuntimeError("missing/small "+name)
    sizes[name]=p.stat().st_size
gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
(OUT/"kaggle_gpu_report.json").write_text(json.dumps({"gpu":gpu,"files":sizes},indent=2)+"\n")
print("EXPORT_DONE",json.dumps({"gpu":gpu,"files":sizes}),flush=True)
'''

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--owner",required=True); ap.add_argument("--out",default="kernel_ironman_export")
    ns=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    scene=(root/"quick_jobs/ironman_nanotech_export/blender_scene_export.py").read_bytes()
    kernel=TEMPLATE.replace("__URL__",repr(BLENDER_URL)).replace("__SCENE__",repr(base64.b64encode(scene).decode()))
    out=root/ns.out; out.mkdir(parents=True,exist_ok=True)
    (out/"kernel.py").write_text(kernel)
    meta={
      "id":f"{ns.owner}/{SLUG}",
      "title":"Ironman Nanotech Export",
      "code_file":"kernel.py","language":"python","kernel_type":"script",
      "is_private":True,"enable_gpu":True,"enable_internet":True,
      "machine_shape":"NvidiaTeslaT4",
      "dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta))
if __name__=="__main__": main()
