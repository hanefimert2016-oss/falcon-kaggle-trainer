#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


SOURCES = {
    "main": {
        "repo": "codelion/fineweb-edu-100M",
        "license": "odc-by",
        "role": "general educational pretraining",
    },
    "coder_python": {
        "repo": "Nan-Do/code-search-net-python",
        "license": "apache-2.0",
        "role": "Python code and documentation",
    },
    "coder_unreal": {
        "repo": "AdamCodd/unreal-engine-5-code",
        "license": "apache-2.0",
        "role": "Unreal Engine 5 API/code knowledge",
    },
    "gui_grounding": {
        "repo": "Fhrozen/GroundCUA",
        "license": "mit",
        "role": "87 desktop software categories with UI boxes/text",
    },
    "gui_actions": {
        "repo": "markov-ai/computer-use",
        "license": "apache-2.0",
        "role": "real desktop screenshot/action trajectories",
    },
    "gui_grounding_extra": {
        "repo": "Salesforce/grounding_dataset",
        "license": "mixed-permissive-and-cc-by",
        "role": "GUI instruction/bbox grounding",
    },
    "minecraft_actions": {
        "repo": "TESS-Computer/minecraft-vla-stage1",
        "license": "mit",
        "role": "Minecraft frame/action pairs",
    },
    "game_visuals": {
        "repo": "Bingsu/Gameplay_Images",
        "license": "cc-by-4.0",
        "role": "10 popular game screen categories including Forza/Minecraft",
    },
    "professional_video": {
        "repo": "markov-ai/computer-use-large",
        "license": "cc-by-4.0",
        "role": "AutoCAD/Blender/Excel/Photoshop/Salesforce/VSCode video frames",
    },
}


def append_bytes(path: Path, payload: bytes) -> int:
    with path.open("ab") as f:
        f.write(payload)
    return len(payload)


def prepare_main(out: Path, target_bytes: int) -> dict:
    path = out / "main_train.bin"
    path.unlink(missing_ok=True)
    ds = load_dataset(SOURCES["main"]["repo"], split="train", streaming=True)
    total = docs = 0
    for row in ds:
        text = str(row.get("text") or "").strip()
        if len(text) < 200:
            continue
        payload = (text + "\n\n<|document|>\n\n").encode("utf-8", "ignore")
        remain = target_bytes - total
        if remain <= 0:
            break
        if len(payload) > remain:
            payload = payload[:remain]
        total += append_bytes(path, payload)
        docs += 1
        if total >= target_bytes:
            break
    if total < min(target_bytes, 32_000_000):
        raise RuntimeError(f"main corpus underfilled: {total} bytes")
    return {"bytes": total, "documents": docs, "file": path.name}


def _coder_record(instruction: str, code: str, source: str) -> bytes:
    text = (
        f"<source>{source}</source>\n"
        f"<instruction>\n{instruction.strip()}\n</instruction>\n"
        f"<code>\n{code.strip()}\n</code>\n\n"
    )
    return text.encode("utf-8", "ignore")


def prepare_coder(out: Path, target_bytes: int) -> dict:
    path = out / "coder_train.bin"
    path.unlink(missing_ok=True)
    total = 0
    counts = {"unreal_engine": 0, "codesearchnet_python": 0}

    # Game-engine knowledge: use a bounded Unreal Engine slice first.
    unreal_budget = min(16_000_000, target_bytes // 4)
    try:
        ds = load_dataset(SOURCES["coder_unreal"]["repo"], split="train", streaming=True)
        for row in ds:
            code = str(row.get("code") or "").strip()
            desc = str(row.get("description") or row.get("className") or "").strip()
            if len(code) < 40:
                continue
            payload = _coder_record(desc or "Explain this Unreal Engine API/code.", code, "unreal-engine-5")
            if total + len(payload) > unreal_budget:
                break
            total += append_bytes(path, payload)
            counts["unreal_engine"] += 1
    except Exception as exc:
        print(f"optional Unreal data failed: {type(exc).__name__}: {exc}", flush=True)

    ds = load_dataset(SOURCES["coder_python"]["repo"], split="train", streaming=True)
    for row in ds:
        code = str(
            row.get("code")
            or row.get("whole_func_string")
            or row.get("func_code_string")
            or ""
        ).strip()
        desc = str(
            row.get("summary")
            or row.get("docstring")
            or row.get("func_documentation_string")
            or ""
        ).strip()
        if len(code) < 80:
            continue
        payload = _coder_record(
            desc or "Explain, complete, or improve this Python code.",
            code,
            "codesearchnet-python",
        )
        remain = target_bytes - total
        if remain <= 0:
            break
        if len(payload) > remain:
            payload = payload[:remain]
        total += append_bytes(path, payload)
        counts["codesearchnet_python"] += 1
        if total >= target_bytes:
            break
    if total < min(target_bytes, 16_000_000):
        raise RuntimeError(f"coder corpus underfilled: {total} bytes")
    return {"bytes": total, "records": counts, "file": path.name}


class ZipShardWriter:
    def __init__(self, root: Path, max_images: int = 1000):
        self.root = root
        self.max_images = max_images
        self.index = -1
        self.count_in_shard = 0
        self.total_images = 0
        self.total_bytes = 0
        self.zf: zipfile.ZipFile | None = None
        self.archive_name = ""

    def _rotate(self):
        if self.zf is not None:
            self.zf.close()
        self.index += 1
        self.count_in_shard = 0
        self.archive_name = f"computer_images_{self.index:03d}.zip"
        self.zf = zipfile.ZipFile(
            self.root / self.archive_name,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=5,
        )

    def add(self, image: Image.Image, prefix: str) -> tuple[str, str, int, int]:
        if self.zf is None or self.count_in_shard >= self.max_images:
            self._rotate()
        image = image.convert("RGB")
        w, h = image.size
        image.thumbnail((768, 768))
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=78, optimize=True)
        payload = buf.getvalue()
        digest = hashlib.sha1(payload).hexdigest()[:16]
        member = f"{prefix}_{self.total_images:07d}_{digest}.jpg"
        assert self.zf is not None
        self.zf.writestr(member, payload)
        self.count_in_shard += 1
        self.total_images += 1
        self.total_bytes += len(payload)
        return self.archive_name, member, w, h

    def close(self):
        if self.zf is not None:
            self.zf.close()
            self.zf = None


def norm_bbox(box, w, h):
    if not box or len(box) != 4:
        return None
    vals = [float(x) for x in box]
    if max(abs(x) for x in vals) <= 1.5:
        x1, y1, x2, y2 = vals
    else:
        x1, y1, x2, y2 = vals[0] / w, vals[1] / h, vals[2] / w, vals[3] / h
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2]


def coord_from_bbox(b):
    return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]


def action_coord(action: str, w: int, h: int):
    m = re.search(r"(?:click|doubleClick|moveTo)\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)", action)
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    if x > 1.0 or y > 1.0:
        x, y = x / max(1, w), y / max(1, h)
    return [max(0.0, min(1.0, x)), max(0.0, min(1.0, y))]


def classify_action(action: str):
    a = action.lower().replace(" ", "")
    if "click(" in a or "doubleclick(" in a:
        return "CLICK"
    if "write(" in a or "typewrite(" in a:
        return "TYPE"
    if "hotkey(" in a or "press(" in a or "keydown(" in a or "keyup(" in a:
        return "KEY"
    if "scroll(" in a or "hscroll(" in a:
        return "SCROLL"
    if "dragto(" in a or "dragrel(" in a:
        return "DRAG"
    if "moveto(" in a or "move(" in a:
        return "MOVE"
    return "OTHER"


def record(
    archive, image, task, action, operation, domain, source,
    *, coord=None, bbox=None, op_valid=False, action_valid=False,
    coord_valid=False, bbox_valid=False, domain_valid=True,
):
    return {
        "archive": archive,
        "image": image,
        "task": str(task)[:2000],
        "action": str(action)[:1000],
        "operation": operation,
        "domain": domain,
        "source": source,
        "coord": coord,
        "bbox": bbox,
        "op_valid": bool(op_valid),
        "action_valid": bool(action_valid),
        "coord_valid": bool(coord_valid),
        "bbox_valid": bool(bbox_valid),
        "domain_valid": bool(domain_valid),
    }


def add_groundcua(rows, writer, target):
    configs = get_dataset_config_names(SOURCES["gui_grounding"]["repo"])
    if os.environ.get("FLM_V05_FAST_PREP") == "1":
        preferred = [
            "Blender","VSCode","PyCharm","IntelliJ_IDEA","Eclipse","NetBeans",
            "Arduino_IDE","Code_Blocks","Qt_Creator","KDevelop","RStudio","Spyder",
            "GIMP","Inkscape","Krita","FreeCAD","Darktable","Lightworks","OpenShot",
            "OpenToonz","Natron","OBS_Studio","LibreOffice_Writer","LibreOffice_Calc",
            "LibreOffice_Impress","OnlyOffice_Document_Editor","OnlyOffice_Spreadsheet",
            "Mozilla_Firefox","Chromium","Brave","Ubuntu_Terminal","Bash","Nemo",
            "VLC_Media_Player","WordPress","QGIS","GrassGIS","Audacity","MuseScore","draw.io",
        ]
        present = set(configs)
        configs = [x for x in preferred if x in present]
    print(f"GroundCUA configs_selected={len(configs)}", flush=True)
    per = max(1, math.ceil(target / max(1, len(configs))))
    added = 0
    category_counts = {}
    for config in configs:
        local = 0
        try:
            ds = load_dataset(SOURCES["gui_grounding"]["repo"], config, split="train", streaming=True)
            for row in ds:
                image = row.get("image")
                boxes = list(row.get("bbox") or [])
                texts = list(row.get("text") or [])
                if image is None or not boxes:
                    continue
                archive, member, w, h = writer.add(image, "ground")
                for i, box in enumerate(boxes):
                    nb = norm_bbox(box, w, h)
                    if nb is None:
                        continue
                    label = texts[i] if i < len(texts) else "UI element"
                    coord = coord_from_bbox(nb)
                    rows.append(record(
                        archive, member,
                        f'Locate and interact with the UI element "{label}".',
                        f"CLICK {coord[0]:.4f} {coord[1]:.4f}",
                        "CLICK", f"app:{config}", "GroundCUA",
                        coord=coord, bbox=nb,
                        op_valid=True, action_valid=True,
                        coord_valid=True, bbox_valid=True,
                    ))
                    added += 1
                    local += 1
                    category_counts[config] = category_counts.get(config, 0) + 1
                    if local >= per or added >= target:
                        break
                if local >= per or added >= target:
                    break
        except Exception as exc:
            print(f"GroundCUA config skip {config}: {type(exc).__name__}: {exc}", flush=True)
        print(f"GroundCUA progress config={config} added={added}/{target}", flush=True)
        if added >= target:
            break
    return {"examples": added, "categories": category_counts, "configs_seen": len(category_counts)}


def add_salesforce(rows, writer, target):
    ds = load_dataset(SOURCES["gui_grounding_extra"]["repo"], split="train", streaming=True)
    added = 0
    for row in ds:
        image = row.get("image")
        if image is None:
            continue
        archive, member, w, h = writer.add(image, "sfdc")
        nb = norm_bbox(row.get("bbox"), w, h)
        if nb is None:
            continue
        coord = coord_from_bbox(nb)
        task = row.get("instruction") or row.get("combine") or row.get("description") or "Interact with the requested GUI element."
        domain = f"gui:{row.get('dataset') or 'salesforce-grounding'}"
        rows.append(record(
            archive, member, task,
            f"CLICK {coord[0]:.4f} {coord[1]:.4f}",
            "CLICK", domain, "Salesforce/grounding_dataset",
            coord=coord, bbox=nb,
            op_valid=True, action_valid=True, coord_valid=True, bbox_valid=True,
        ))
        added += 1
        if added >= target:
            break
    return {"examples": added}


def add_markov_actions(rows, writer, target):
    ds = load_dataset(SOURCES["gui_actions"]["repo"], split="train", streaming=True)
    added = trajectories = 0
    for row in ds:
        instruction = str(row.get("instruction") or "")
        actions = list(row.get("actions") or [])
        shots = list(row.get("screenshots") or [])
        local = 0
        for image, action in zip(shots, actions):
            if image is None or not action:
                continue
            archive, member, w, h = writer.add(image, "act")
            op = classify_action(str(action))
            coord = action_coord(str(action), w, h)
            rows.append(record(
                archive, member, instruction, action, op,
                f"osworld:{row.get('domain') or 'desktop'}",
                "markov-ai/computer-use",
                coord=coord,
                op_valid=True, action_valid=True,
                coord_valid=coord is not None,
            ))
            added += 1
            local += 1
            if added >= target or local >= 8:
                break
        if local:
            trajectories += 1
        if added >= target:
            break
    return {"examples": added, "trajectories": trajectories}


def add_minecraft(rows, writer, target):
    ds = load_dataset(SOURCES["minecraft_actions"]["repo"], split="train", streaming=True)
    added = 0
    for row in ds:
        raw = row.get("image")
        if not raw:
            continue
        try:
            image = Image.open(io.BytesIO(raw))
        except Exception:
            continue
        archive, member, _, _ = writer.add(image, "mc")
        action = str(row.get("action") or "")
        rows.append(record(
            archive, member,
            "Continue the Minecraft task from the current visual state.",
            action,
            "GAME_ACTION",
            "game:Minecraft",
            "TESS-Computer/minecraft-vla-stage1",
            op_valid=True,
            action_valid=bool(action),
        ))
        added += 1
        if added >= target:
            break
    return {"examples": added}


def add_game_visuals(rows, writer, target):
    ds = load_dataset(SOURCES["game_visuals"]["repo"], split="train", streaming=True)
    names = getattr(ds.features.get("label"), "names", None) or []
    added = 0
    counts = {}
    for row in ds:
        image = row.get("image")
        if image is None:
            continue
        label = row.get("label")
        if isinstance(label, int) and label < len(names):
            name = names[label]
        else:
            name = str(label)
        archive, member, _, _ = writer.add(image, "game")
        rows.append(record(
            archive, member,
            "Recognize the game/interface shown on the screen.",
            "",
            "OTHER",
            f"game:{name}",
            "Bingsu/Gameplay_Images",
            domain_valid=True,
        ))
        counts[name] = counts.get(name, 0) + 1
        added += 1
        if added >= target:
            break
    return {"examples": added, "games": counts}


def add_professional_video_frames(rows, writer, frames_per_category=2):
    try:
        import cv2
    except Exception as exc:
        return {"examples": 0, "error": f"opencv unavailable: {exc}"}

    cats = ["autocad", "blender", "excel", "photoshop", "salesforce", "vscode"]
    added = 0
    details = {}
    for cat in cats:
        try:
            meta_path = hf_hub_download(
                SOURCES["professional_video"]["repo"],
                f"data/{cat}/metadata.jsonl",
                repo_type="dataset",
            )
            entries = [
                json.loads(x)
                for x in Path(meta_path).read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            entries.sort(key=lambda x: float(x.get("trimmed_duration") or 1e18))
            entry = entries[0]
            filename = entry["file_name"]
            video_path = hf_hub_download(
                SOURCES["professional_video"]["repo"],
                f"data/{cat}/{filename}",
                repo_type="dataset",
            )
            cap = cv2.VideoCapture(video_path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            local = 0
            for frac in (0.25, 0.6, 0.85)[:frames_per_category]:
                if n > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * frac)))
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                archive, member, _, _ = writer.add(image, "pro")
                rows.append(record(
                    archive, member,
                    "Understand the professional software workspace shown on screen.",
                    "",
                    "OTHER",
                    f"professional:{cat}",
                    "markov-ai/computer-use-large",
                    domain_valid=True,
                ))
                added += 1
                local += 1
            cap.release()
            details[cat] = local
        except Exception as exc:
            details[cat] = f"skip:{type(exc).__name__}:{exc}"
            print(f"professional video skip {cat}: {type(exc).__name__}: {exc}", flush=True)
    return {"examples": added, "categories": details}


def prepare_computer(out: Path, target: int) -> dict:
    rows = []
    writer = ZipShardWriter(out, max_images=750)
    # Allocate so total is >= target even if the optional video slice fails.
    allocations = {
        "groundcua": int(target * 0.58),
        "salesforce": int(target * 0.16),
        "markov_actions": int(target * 0.10),
        "minecraft": int(target * 0.08),
        "games": int(target * 0.08),
    }
    stats = {}
    try:
        stats["groundcua"] = add_groundcua(rows, writer, allocations["groundcua"])
        print(f"v05_data groundcua_done rows={len(rows)}", flush=True)
        stats["salesforce"] = add_salesforce(rows, writer, allocations["salesforce"])
        print(f"v05_data salesforce_done rows={len(rows)}", flush=True)
        stats["markov_actions"] = add_markov_actions(rows, writer, allocations["markov_actions"])
        print(f"v05_data markov_done rows={len(rows)}", flush=True)
        stats["minecraft"] = add_minecraft(rows, writer, allocations["minecraft"])
        print(f"v05_data minecraft_done rows={len(rows)}", flush=True)
        stats["games"] = add_game_visuals(rows, writer, allocations["games"])
        print(f"v05_data games_done rows={len(rows)}", flush=True)
        if os.environ.get("FLM_V05_FAST_PREP") == "1":
            stats["professional_video"] = {"examples": 0, "skipped": "pilot_fast_prep"}
        else:
            stats["professional_video"] = add_professional_video_frames(rows, writer, 2)
        print(f"v05_data professional_done rows={len(rows)}", flush=True)
    finally:
        writer.close()

    manifest = out / "computer_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    domains = sorted({r["domain"] for r in rows})
    sources = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1

    if len(rows) < int(target * 0.90):
        raise RuntimeError(f"computer-use data underfilled: {len(rows)} < {target}")
    if len(domains) < 30:
        raise RuntimeError(f"computer-use domain diversity too low: {len(domains)}")

    return {
        "examples": len(rows),
        "unique_images": writer.total_images,
        "image_bytes": writer.total_bytes,
        "domains": len(domains),
        "domain_names": domains,
        "source_counts": sources,
        "parts": stats,
        "manifest": manifest.name,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prepared_v05")
    ap.add_argument("--owner", default=os.environ.get("KAGGLE_OWNER", "owner"))
    ap.add_argument("--main-bytes", type=int, default=128_000_000)
    ap.add_argument("--coder-bytes", type=int, default=64_000_000)
    ap.add_argument("--computer-examples", type=int, default=6000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"v05_data main_start target={args.main_bytes}", flush=True)
    main_stats = prepare_main(out, args.main_bytes)
    print(f"v05_data main_done {main_stats}", flush=True)
    print(f"v05_data coder_start target={args.coder_bytes}", flush=True)
    coder_stats = prepare_coder(out, args.coder_bytes)
    print(f"v05_data coder_done {coder_stats}", flush=True)
    print(f"v05_data computer_start target={args.computer_examples}", flush=True)
    computer_stats = prepare_computer(out, args.computer_examples)
    print(f"v05_data computer_done examples={computer_stats['examples']} domains={computer_stats['domains']}", flush=True)
    stats = {
        "main": main_stats,
        "coder": coder_stats,
        "computer_use": computer_stats,
    }
    manifest = {
        "pipeline_version": "v0.5",
        "phase": "CPU download/preprocess; GPU reserved for train/eval",
        "sources": SOURCES,
        "stats": stats,
    }
    (out / "sources.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Falcon FLM HF Training Data v05",
                "id": f"{args.owner}/flm-hf-v05",
                "licenses": [{"name": "other"}],
                "description": "CPU-prepared multi-source FLM v0.5 corpora. Upstream source licenses and counts are recorded in sources.json.",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
