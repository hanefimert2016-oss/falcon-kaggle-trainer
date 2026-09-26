#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import zipfile

from datasets import load_dataset
from PIL import Image

SOURCES = {
    "main": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "license": "odc-by",
    },
    "coder": {
        "repo": "Nan-Do/code-search-net-python",
        "config": None,
        "license": "apache-2.0",
    },
    "computer_use": {
        "repo": "osunlp/Multimodal-Mind2Web",
        "config": None,
        "license": "openrail",
    },
}

def prepare_main(out: Path, max_docs: int, max_bytes: int) -> dict:
    ds = load_dataset(
        SOURCES["main"]["repo"],
        SOURCES["main"]["config"],
        split="train",
        streaming=True,
    )
    total = 0
    count = 0
    with (out / "main_train.txt").open("w", encoding="utf-8") as f:
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
    return {"examples": count, "bytes": total}

def prepare_coder(out: Path, max_examples: int, max_bytes: int) -> dict:
    ds = load_dataset(SOURCES["coder"]["repo"], split="train", streaming=True)
    total = 0
    count = 0
    with (out / "coder_train.jsonl").open("w", encoding="utf-8") as f:
        for row in ds:
            code = str(row.get("code") or "").strip()
            doc = str(row.get("docstring") or "").strip()
            if len(code) < 80:
                continue
            record = {
                "instruction": doc[:4000] or "Complete or explain the following code.",
                "code": code[:20000],
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
    return {"examples": count, "bytes": total}

def normalize_op(value) -> str:
    if isinstance(value, dict):
        return str(value.get("op") or value.get("original_op") or "CLICK").upper()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return str(
                    parsed.get("op") or parsed.get("original_op") or "CLICK"
                ).upper()
        except Exception:
            upper = value.upper()
            for op in ("CLICK", "TYPE", "SELECT"):
                if op in upper:
                    return op
    return "CLICK"

def prepare_computer(out: Path, max_examples: int) -> dict:
    ds = load_dataset(
        SOURCES["computer_use"]["repo"],
        split="train",
        streaming=True,
    )
    image_bytes = 0
    manifest = []

    with zipfile.ZipFile(
        out / "computer_images.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for row in ds:
            image = row.get("screenshot")
            if not isinstance(image, Image.Image):
                continue
            task = str(row.get("confirmed_task") or "").strip()
            action = str(row.get("target_action_reprs") or "").strip()
            if not task or not action:
                continue

            image = image.convert("RGB")
            image.thumbnail((640, 640))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=82, optimize=True)
            payload = buf.getvalue()
            name = f"{len(manifest):05d}.jpg"
            zf.writestr(name, payload)
            image_bytes += len(payload)

            manifest.append(
                {
                    "image": name,
                    "task": task[:4000],
                    "action": action[:1000],
                    "operation": normalize_op(row.get("operation")),
                    "website": str(row.get("website") or ""),
                }
            )
            if len(manifest) >= max_examples:
                break

    with (out / "computer_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"examples": len(manifest), "image_bytes": image_bytes}

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="prepared_hf")
    p.add_argument("--owner", default=os.environ.get("KAGGLE_OWNER", "owner"))
    p.add_argument("--main-docs", type=int, default=128)
    p.add_argument("--main-bytes", type=int, default=3_000_000)
    p.add_argument("--coder-examples", type=int, default=600)
    p.add_argument("--coder-bytes", type=int, default=3_000_000)
    p.add_argument("--computer-examples", type=int, default=32)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = {
        "main": prepare_main(out, args.main_docs, args.main_bytes),
        "coder": prepare_coder(out, args.coder_examples, args.coder_bytes),
        "computer_use": prepare_computer(out, args.computer_examples),
    }
    source_manifest = {"sources": SOURCES, "stats": stats}
    (out / "sources.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Falcon FLM HF Training Data v04",
                "id": f"{args.owner}/flm-hf-v04",
                "licenses": [{"name": "other"}],
                "description": (
                    "Private CPU-prepared shard from FineWeb-Edu, "
                    "CodeSearchNet-Python and Multimodal-Mind2Web. "
                    "See sources.json for original source licenses."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(source_manifest, indent=2, ensure_ascii=False))
    if any(stats[name]["examples"] == 0 for name in stats):
        raise SystemExit("one or more Hugging Face sources produced no usable samples")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
