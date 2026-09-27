#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import zipfile
from pathlib import Path


def payload(root: Path) -> str:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root / "flm").rglob("*.py")):
            z.writestr(p.relative_to(root).as_posix(), p.read_bytes())
    return base64.b64encode(b.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--out", default="kernel_v07_dev_computer_cpu")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    src = (root / "flm/data/prepare_v07_computer_dev.py").read_text(encoding="utf-8")
    p = payload(root)
    chunks = "\n".join(f'    "{p[i:i+100]}"' for i in range(0, len(p), 100))
    wrapper = f'''# Auto-generated v0.7 DEV ComputerUse CPU data kernel.
import base64,os,shutil,subprocess,sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT","180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT","45")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")

print("V07_DEV_COMPUTER_BOOT",flush=True)
print("python",sys.version,flush=True)
print("disk_before",shutil.disk_usage("/kaggle/working"),flush=True)
subprocess.check_call([
    sys.executable,"-m","pip","install","-q",
    "datasets>=4.0,<5","huggingface_hub>=0.30,<2","pillow>=10"
])
_PAYLOAD=(
{chunks}
)
z=Path("/kaggle/working/flm_v07_dev_package.zip")
z.write_bytes(base64.b64decode(_PAYLOAD))
sys.path.insert(0,str(z))
exec(compile({json.dumps(src)},"prepare_v07_computer_dev.py","exec"),{{"__name__":"__main__"}})
'''
    (out / "prepare_computer.py").write_text(wrapper, encoding="utf-8")
    meta = {
        "id": f"{args.owner}/falcon-flm-v07-dev-computer-cpu",
        "title": "Falcon FLM v07 DEV ComputerUse CPU Data",
        "code_file": "prepare_computer.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"kernel": meta["id"], "payload_bytes": len(p)}))


if __name__ == "__main__":
    main()
