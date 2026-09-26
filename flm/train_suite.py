#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from flm.train_text import train_text
from flm.train_computer_use import train_computer


EXPECTED_SOURCES = {
    "main": "codelion/fineweb-edu-100M",
    "coder": "Nan-Do/code-search-net-python",
    "computer_use": "markov-ai/computer-use",
}


def main() -> int:
    data_root = Path(
        os.environ.get("FLM_DATA_ROOT", "/kaggle/input/flm-hf-v04")
    )
    output_root = Path(
        os.environ.get("FLM_OUTPUT_ROOT", "/kaggle/working/flm-v0.4")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    sources_path = data_root / "sources.json"
    if not sources_path.is_file():
        raise SystemExit(f"missing real-data source manifest: {sources_path}")
    source_manifest = json.loads(sources_path.read_text(encoding="utf-8"))
    for name, repo in EXPECTED_SOURCES.items():
        actual = source_manifest.get("sources", {}).get(name, {}).get("repo")
        if actual != repo:
            raise SystemExit(
                f"wrong {name} dataset: expected {repo!r}, got {actual!r}"
            )
    shutil.copy2(sources_path, output_root / "sources.json")

    # Three independently trained checkpoints.
    results = [
        train_text("main", data_root, output_root),
        train_computer(data_root, output_root),
        train_text("coder", data_root, output_root),
    ]
    summary = {
        "models": results,
        "data_phase": "CPU/HuggingFace",
        "train_eval_phase": results[0]["accelerator"],
    }
    (output_root / "suite_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"suite_complete output={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
