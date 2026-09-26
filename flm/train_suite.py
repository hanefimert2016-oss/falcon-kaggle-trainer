#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from flm.train_text import train_text
from flm.train_computer_use import train_computer

def main() -> int:
    data_root = Path(
        os.environ.get("FLM_DATA_ROOT", "/kaggle/input/flm-hf-v04")
    )
    output_root = Path(
        os.environ.get("FLM_OUTPUT_ROOT", "/kaggle/working/flm-v0.4")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    results = [
        train_text("main", data_root, output_root),
        train_computer(data_root, output_root),
        train_text("coder", data_root, output_root),
    ]
    (output_root / "suite_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"suite_complete output={output_root}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
