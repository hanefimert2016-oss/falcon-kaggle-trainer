#!/usr/bin/env python3
"""FLM v0.7 strict zero-train Colab helper.

Neural model training was retired from the authoritative v0.7 path. The only
learned artifact is the tokenizer, and that is produced by the Kaggle
falcon-flm-v07-tokenizer-only job. Colab may run/evaluate the Core, but it does
not train a model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--memory")
    ap.add_argument("--ingest",action="append",default=[])
    ap.add_argument("--prompt",default="Sen kimsin?")
    args=ap.parse_args()

    root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable,str(root/"scripts/run_zero_train_v07.py"),"--prompt",args.prompt]
    if args.memory:
        cmd += ["--memory",args.memory]
    for item in args.ingest:
        cmd += ["--ingest",item]
    print("FLM v0.7: neural_model_training=false; tokenizer_training=Kaggle-only",flush=True)
    return subprocess.run(cmd,cwd=root).returncode


if __name__=="__main__":
    raise SystemExit(main())
