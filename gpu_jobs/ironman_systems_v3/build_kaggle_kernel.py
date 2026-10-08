#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, json
from pathlib import Path

BLENDER_URL='https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz'
SLUG='iron-man-systems-lab-v4'

KERNEL=r'''import base64, json, os, pathlib, shutil, subprocess, tarfile, urllib.request
OUT=pathlib.Path('/kaggle/working'); OUT.mkdir(parents=True,exist_ok=True)
print('=== GPU ===',flush=True); subprocess.run(['nvidia-smi'],check=False)
url=__URL__; arc=OUT/'blender.tar.xz'; bd=OUT/'blender'
if not bd.exists():
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 IronmanSystemsLab/3.0'})
    with urllib.request.urlopen(req,timeout=180) as src, arc.open('wb') as dst: shutil.copyfileobj(src,dst,1024*1024)
    with tarfile.open(arc,'r:xz') as tf: tf.extractall(OUT)
    dirs=sorted(OUT.glob('blender-4.5.*-linux-x64'))
    if not dirs: raise RuntimeError('Blender directory not found')
    dirs[0].rename(bd)
blender=bd/'blender'; blender.chmod(0o755)
scene=OUT/'blender_systems_v3.py'; scene.write_bytes(base64.b64decode(__SCENE__))
env=os.environ.copy(); env['IRONMAN_OUT']=str(OUT)
cmd=[str(blender),'-b','--python',str(scene)]
print('+',' '.join(cmd),flush=True); subprocess.check_call(cmd,env=env)
expected=['ironman_systems_v3_editable.blend','ironman_systems_v3_interactive.glb','ironman_systems_v3_poster.png','ironman_systems_v3_presentation.mp4','ironman_systems_v3_manifest.json']
sizes={}
for name in expected:
    p=OUT/name
    if not p.exists() or p.stat().st_size<1024: raise RuntimeError('missing/small '+name)
    sizes[name]=p.stat().st_size
for p in [scene,arc]:
    try:
        if p.exists(): p.unlink()
    except Exception as e: print('cleanup',p,e,flush=True)
try:
    if bd.exists(): shutil.rmtree(bd)
except Exception as e: print('cleanup blender',e,flush=True)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
report={'gpu':gpu,'files':sizes,'blender_url':url}
(OUT/'ironman_systems_v3_gpu_report.json').write_text(json.dumps(report,indent=2)+'\n')
print('KAGGLE_IRONMAN_SYSTEMS_V3_DONE',json.dumps(report),flush=True)
'''

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--owner',required=True); ap.add_argument('--out',default='kernel_ironman_systems_v3')
    ns=ap.parse_args(); root=Path(__file__).resolve().parents[2]
    scene=(root/'gpu_jobs/ironman_systems_v3/blender_systems_v3.py').read_bytes()
    kernel=KERNEL.replace('__URL__',repr(BLENDER_URL)).replace('__SCENE__',repr(base64.b64encode(scene).decode('ascii')))
    out=root/ns.out; out.mkdir(parents=True,exist_ok=True)
    (out/'kernel.py').write_text(kernel,encoding='utf-8')
    meta={
      'id':f'{ns.owner}/{SLUG}','title':'Iron Man Systems Lab v4','code_file':'kernel.py','language':'python','kernel_type':'script',
      'is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
      'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]
    }
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'kernel':meta['id'],'scene_bytes':len(scene),'out':str(out)}))
if __name__=='__main__': main()
