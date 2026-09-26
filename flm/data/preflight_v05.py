#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from datasets import Image as HFImage, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download


def first_row(repo: str, config: str | None = None, decode_images: bool = True):
    ds = load_dataset(repo, config, split="train", streaming=True)
    if not decode_images and "image" in ds.features:
        ds = ds.cast_column("image", HFImage(decode=False))
    row = next(iter(ds))
    return {
        "keys": sorted(row.keys()),
        "types": {k: type(v).__name__ for k, v in row.items()},
        "features": str(ds.features),
    }


def safe_probe(name: str, fn, report: dict):
    try:
        value = fn()
        report[name] = {"ok": True, "data": value}
        print(f"PRECHECK_OK {name}", flush=True)
        return True
    except Exception as exc:
        report[name] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(f"PRECHECK_OPTIONAL_FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        return False


def main() -> int:
    report: dict[str, object] = {"phase": "v0.5_cpu_preflight"}

    ground_configs = get_dataset_config_names("Fhrozen/GroundCUA")
    report["groundcua"] = {
        "ok": True,
        "data": {
            "config_count": len(ground_configs),
            "configs": ground_configs,
            "sample": first_row("Fhrozen/GroundCUA", ground_configs[0]),
        },
    }
    print(f"PRECHECK_OK groundcua configs={len(ground_configs)}", flush=True)

    safe_probe(
        "markov_actions",
        lambda: first_row("markov-ai/computer-use"),
        report,
    )
    safe_probe(
        "salesforce_grounding",
        lambda: first_row("Salesforce/grounding_dataset"),
        report,
    )
    safe_probe(
        "minecraft_vla",
        lambda: first_row("TESS-Computer/minecraft-vla-stage1"),
        report,
    )
    # This source has occasionally shipped broken paths inside its image archive.
    # Keep it optional and inspect metadata without decoding pixels.
    safe_probe(
        "scenario_recognition",
        lambda: first_row(
            "amazingtrash/scenario-recognition-for-display",
            decode_images=False,
        ),
        report,
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
    report["professional_video"] = {"ok": True, "data": pro_meta}
    print("PRECHECK_OK professional_video", flush=True)

    Path("v05_preflight_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)

    required = ("groundcua", "markov_actions", "salesforce_grounding", "minecraft_vla")
    failed = [name for name in required if not report.get(name, {}).get("ok")]
    if failed:
        raise SystemExit(f"required v0.5 sources failed preflight: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
