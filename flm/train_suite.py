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


def _valid_manifest(path: Path) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for name, repo in EXPECTED_SOURCES.items():
        if manifest.get("sources", {}).get(name, {}).get("repo") != repo:
            return False
    return True


def resolve_data_root() -> Path:
    configured = os.environ.get("FLM_DATA_ROOT", "").strip()
    candidates: list[Path] = []

    if configured:
        root = Path(configured)
        direct = root / "sources.json"
        if direct.is_file():
            candidates.append(direct)
        if root.exists():
            candidates.extend(root.rglob("sources.json"))

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.rglob("sources.json"))

    seen: set[Path] = set()
    for manifest in candidates:
        manifest = manifest.resolve()
        if manifest in seen:
            continue
        seen.add(manifest)
        if _valid_manifest(manifest):
            print(f"resolved_data_root={manifest.parent}", flush=True)
            return manifest.parent

    searched = [configured] if configured else []
    searched.append("/kaggle/input/**/sources.json")
    raise SystemExit(
        "could not locate valid real-data source manifest; searched: "
        + ", ".join(searched)
    )


def main() -> int:
    data_root = resolve_data_root()
    output_root = Path(
        os.environ.get("FLM_OUTPUT_ROOT", "/kaggle/working/flm-v0.4")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    sources_path = data_root / "sources.json"
    source_manifest = json.loads(sources_path.read_text(encoding="utf-8"))
    for name, repo in EXPECTED_SOURCES.items():
        actual = source_manifest.get("sources", {}).get(name, {}).get("repo")
        if actual != repo:
            raise SystemExit(
                f"wrong {name} dataset: expected {repo!r}, got {actual!r}"
            )
    shutil.copy2(sources_path, output_root / "sources.json")

    print(
        json.dumps(
            {
                "resolved_data_root": str(data_root),
                "output_root": str(output_root),
                "files": sorted(p.name for p in data_root.iterdir())[:50],
            }
        ),
        flush=True,
    )

    results = [
        train_text("main", data_root, output_root),
        train_computer(data_root, output_root),
        train_text("coder", data_root, output_root),
    ]
    summary = {
        "models": results,
        "data_phase": "CPU/HuggingFace",
        "train_eval_phase": results[0]["accelerator"],
        "resolved_data_root": str(data_root),
    }
    (output_root / "suite_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"suite_complete output={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
