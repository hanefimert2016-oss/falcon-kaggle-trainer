#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
import zipfile

# This process is intentionally CPU-only. GPU/TPU is reserved for train/eval.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from datasets import load_dataset
from PIL import Image

SOURCES = {
    "main": {
        "repo": "codelion/fineweb-edu-100M",
        "config": None,
        "license": "odc-by",
        "purpose": "general/educational language pretraining",
    },
    "coder": {
        "repo": "Nan-Do/code-search-net-python",
        "config": None,
        "license": "apache-2.0",
        "purpose": "code + natural-language programming supervision",
    },
    "computer_use": {
        "repo": "markov-ai/computer-use",
        "config": None,
        "license": "apache-2.0",
        "purpose": "desktop screenshots + task instructions + UI actions",
    },
}

OPS = ("CLICK", "TYPE", "KEY", "SCROLL", "MOVE", "OTHER")


def _load_image(value) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload:
            try:
                return Image.open(io.BytesIO(payload))
            except Exception:
                return None
        path = value.get("path")
        if path:
            try:
                return Image.open(path)
            except Exception:
                return None
    return None


def classify_action(action: str) -> str:
    a = action.lower().replace(" ", "")
    if "click(" in a or "doubleclick(" in a:
        return "CLICK"
    if "write(" in a or "typewrite(" in a:
        return "TYPE"
    if "hotkey(" in a or "press(" in a or "keydown(" in a or "keyup(" in a:
        return "KEY"
    if "scroll(" in a or "hscroll(" in a:
        return "SCROLL"
    if "moveto(" in a or "move(" in a or "dragto(" in a or "dragrel(" in a:
        return "MOVE"
    return "OTHER"


def prepare_main(out: Path, max_docs: int, max_bytes: int) -> dict:
    ds = load_dataset(SOURCES["main"]["repo"], split="train", streaming=True)
    total = 0
    count = 0
    path = out / "main_train.txt"
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = str(row.get("text") or "").strip()
            if len(text) < 200:
                continue
            encoded = text.encode("utf-8", "ignore")
            if total + len(encoded) > max_bytes and count:
                break
            f.write(text)
            f.write("\n\n<|document|>\n\n")
            total += len(encoded)
            count += 1
            if count >= max_docs or total >= max_bytes:
                break
    return {"examples": count, "bytes": total, "file": path.name}


def prepare_coder(out: Path, max_examples: int, max_bytes: int) -> dict:
    ds = load_dataset(SOURCES["coder"]["repo"], split="train", streaming=True)
    total = 0
    count = 0
    path = out / "coder_train.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            code = str(
                row.get("code")
                or row.get("whole_func_string")
                or row.get("func_code_string")
                or ""
            ).strip()
            doc = str(
                row.get("summary")
                or row.get("docstring")
                or row.get("func_documentation_string")
                or ""
            ).strip()
            if len(code) < 80:
                continue
            record = {
                "instruction": doc[:4000]
                or "Explain, complete, or improve the following Python code.",
                "code": code[:24000],
            }
            line = json.dumps(record, ensure_ascii=False)
            size = len(line.encode("utf-8"))
            if total + size > max_bytes and count:
                break
            f.write(line + "\n")
            total += size
            count += 1
            if count >= max_examples or total >= max_bytes:
                break
    return {"examples": count, "bytes": total, "file": path.name}


def prepare_computer(
    out: Path,
    max_examples: int,
    max_trajectories: int,
    max_steps_per_trajectory: int,
) -> dict:
    ds = load_dataset(
        SOURCES["computer_use"]["repo"],
        split="train",
        streaming=True,
    )
    manifest = []
    image_bytes = 0
    trajectories = 0
    op_counts = {name: 0 for name in OPS}

    zip_path = out / "computer_images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in ds:
            instruction = str(row.get("instruction") or "").strip()
            actions = list(row.get("actions") or [])
            screenshots = list(row.get("screenshots") or [])
            if not instruction or not actions or not screenshots:
                continue

            usable = min(len(actions), len(screenshots), max_steps_per_trajectory)
            added_this_trajectory = 0
            for step in range(usable):
                image = _load_image(screenshots[step])
                action = str(actions[step] or "").strip()
                if image is None or not action:
                    continue

                image = image.convert("RGB")
                image.thumbnail((640, 640))
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=80, optimize=True)
                payload = buf.getvalue()
                name = f"{len(manifest):05d}.jpg"
                zf.writestr(name, payload)
                image_bytes += len(payload)

                operation = classify_action(action)
                op_counts[operation] += 1
                manifest.append(
                    {
                        "image": name,
                        "task": instruction[:4000],
                        "action": action[:1200],
                        "operation": operation,
                        "domain": str(row.get("domain") or ""),
                        "task_id": str(row.get("task_id") or ""),
                        "step": step,
                    }
                )
                added_this_trajectory += 1
                if len(manifest) >= max_examples:
                    break

            if added_this_trajectory:
                trajectories += 1
            if len(manifest) >= max_examples or trajectories >= max_trajectories:
                break

    path = out / "computer_manifest.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "examples": len(manifest),
        "trajectories": trajectories,
        "image_bytes": image_bytes,
        "operations": op_counts,
        "manifest": path.name,
        "images": zip_path.name,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="CPU-only preparation of real Hugging Face training shards."
    )
    p.add_argument("--out", default="prepared_hf")
    p.add_argument("--owner", default=os.environ.get("KAGGLE_OWNER", "owner"))
    p.add_argument("--main-docs", type=int, default=256)
    p.add_argument("--main-bytes", type=int, default=8_000_000)
    p.add_argument("--coder-examples", type=int, default=800)
    p.add_argument("--coder-bytes", type=int, default=8_000_000)
    p.add_argument("--computer-examples", type=int, default=48)
    p.add_argument("--computer-trajectories", type=int, default=12)
    p.add_argument("--computer-steps", type=int, default=6)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = {
        "main": prepare_main(out, args.main_docs, args.main_bytes),
        "coder": prepare_coder(out, args.coder_examples, args.coder_bytes),
        "computer_use": prepare_computer(
            out,
            args.computer_examples,
            args.computer_trajectories,
            args.computer_steps,
        ),
    }
    source_manifest = {
        "phase": "cpu_hf_download_and_preprocessing",
        "sources": SOURCES,
        "stats": stats,
    }
    (out / "sources.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Private Kaggle dataset: mixed upstream licenses are tracked in sources.json.
    (out / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Falcon FLM HF Training Data v04",
                "id": f"{args.owner}/flm-hf-v04",
                "licenses": [{"name": "other"}],
                "description": (
                    "CPU-prepared real Hugging Face shards for Falcon FLM: "
                    "FineWeb-Edu 100M sample, CodeSearchNet Python, and "
                    "Apache-2.0 computer-use screenshot/action trajectories. "
                    "Original source metadata is recorded in sources.json."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(source_manifest, indent=2, ensure_ascii=False), flush=True)
    if any(stats[name].get("examples", 0) == 0 for name in stats):
        raise SystemExit("one or more Hugging Face sources produced no usable samples")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
