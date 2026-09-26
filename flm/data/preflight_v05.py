#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download


def first_row(repo: str, config: str | None = None):
    ds = load_dataset(repo, config, split="train", streaming=True)
    row = next(iter(ds))
    return {
        "keys": sorted(row.keys()),
        "types": {k: type(v).__name__ for k, v in row.items()},
        "features": str(ds.features),
    }


def main() -> int:
    report: dict[str, object] = {"phase": "v0.5_cpu_preflight"}

    ground_configs = get_dataset_config_names("Fhrozen/GroundCUA")
    report["groundcua"] = {
        "config_count": len(ground_configs),
        "configs": ground_configs,
        "sample": first_row("Fhrozen/GroundCUA", ground_configs[0]),
    }

    report["markov_actions"] = first_row("markov-ai/computer-use")
    report["salesforce_grounding"] = first_row("Salesforce/grounding_dataset")
    report["minecraft_vla"] = first_row("TESS-Computer/minecraft-vla-stage1")
    report["scenario_recognition"] = first_row(
        "amazingtrash/scenario-recognition-for-display"
    )

    pro_categories = [
        "autocad",
        "blender",
        "excel",
        "photoshop",
        "salesforce",
        "vscode",
    ]
    pro_meta = {}
    for cat in pro_categories:
        filename = f"data/{cat}/metadata.jsonl"
        path = hf_hub_download(
            "markov-ai/computer-use-large",
            filename,
            repo_type="dataset",
        )
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        pro_meta[cat] = {
            "rows_in_metadata": len(lines),
            "keys": sorted(row.keys()),
            "sample": row,
        }
    report["professional_video"] = pro_meta

    Path("v05_preflight_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
