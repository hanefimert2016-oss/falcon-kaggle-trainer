#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import time

from datasets import load_dataset

from flm.data.prepare_v07 import ImageShardWriter, prepare_rexx
from flm.data.prepare_v05 import (
    ZipShardWriter,
    add_groundcua,
    add_salesforce,
)


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def payload_from_action(raw: str, op: str) -> str:
    try:
        tree = ast.parse(str(raw or ""))
    except Exception:
        return ""
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        f = call.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if op == "TYPE" and name in {"write", "typewrite"} and call.args:
            v = _literal(call.args[0])
            return str(v) if isinstance(v, str) else ""
        if op == "KEY" and name in {"hotkey", "press"}:
            vals = [_literal(x) for x in call.args]
            return "+".join(str(x) for x in vals if isinstance(x, (str, int)))
        if op == "SCROLL" and name in {"scroll", "hscroll"} and call.args:
            v = _literal(call.args[0])
            if isinstance(v, (int, float)):
                return f"{int(v)},0" if name == "hscroll" else f"0,{int(v)}"
    return ""


def _manifest_stats(manifest: Path) -> dict:
    rows = 0
    ops: dict[str, int] = {}
    sources: dict[str, int] = {}
    domains: set[str] = set()
    archives: dict[str, int] = {}
    if not manifest.is_file():
        return {"examples": 0, "ops": {}, "sources": {}, "domains": 0, "archives": {}}
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows += 1
            op = str(r.get("operation") or "OTHER").upper()
            src = str(r.get("source") or "unknown")
            ops[op] = ops.get(op, 0) + 1
            sources[src] = sources.get(src, 0) + 1
            domains.add(str(r.get("domain") or ""))
            archive = str(r.get("archive") or "")
            if archive:
                archives[archive] = archives.get(archive, 0) + 1
    return {
        "examples": rows,
        "ops": ops,
        "sources": sources,
        "domains": len(domains),
        "archives": archives,
    }


def _write_progress(out: Path, stats: dict, stage: str) -> dict:
    manifest = out / "cu07_manifest.jsonl"
    summary = {
        "pipeline_version": "v0.7-dev-computer-partial",
        "stage": stage,
        "manifest": _manifest_stats(manifest),
        "parts": stats,
        "updated_unix": int(time.time()),
    }
    (out / "progress.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "V07_DEV_CU_PROGRESS "
        + json.dumps(
            {
                "stage": stage,
                "examples": summary["manifest"]["examples"],
                "ops": summary["manifest"]["ops"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def _safe_stage(name: str, fn, stats: dict, out: Path, *, required: bool = False):
    started = time.time()
    try:
        result = fn()
        if not isinstance(result, dict):
            result = {"result": result}
        result["elapsed_s"] = round(time.time() - started, 3)
        result["status"] = "ok"
        stats[name] = result
        print(f"V07_DEV_CU_STAGE_OK {name} {json.dumps(result, ensure_ascii=False)}", flush=True)
    except Exception as exc:
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.time() - started, 3),
        }
        stats[name] = result
        print(f"V07_DEV_CU_STAGE_ERROR {name} {result['error']}", flush=True)
        _write_progress(out, stats, name + "_error")
        if required:
            raise
    _write_progress(out, stats, name)
    return stats[name]


def _append_legacy_rows(manifest: Path, rows: list[dict]) -> dict:
    accepted = 0
    ops: dict[str, int] = {}
    with manifest.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            op = str(r.get("operation") or "").upper()
            if op not in {
                "MOVE", "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "SCROLL",
                "TYPE", "KEY", "KEY_DOWN", "KEY_UP",
            }:
                continue
            coord = r.get("coord") if r.get("coord_valid") else None
            if op in {"MOVE", "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"} and coord is None:
                continue
            payload = payload_from_action(r.get("action", ""), op)
            if op in {"TYPE", "KEY", "KEY_DOWN", "KEY_UP", "SCROLL"} and not payload:
                continue
            rec = {
                "archive": r["archive"],
                "image": r["image"],
                "task": r.get("task", ""),
                "operation": op,
                "coord": coord,
                "coord2": None,
                "payload": payload,
                "domain": r.get("domain", "legacy:desktop"),
                "source": r.get("source", "legacy"),
                "episode_id": f"ground:{r.get('source', 'legacy')}:{i}",
                "step": 0,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            accepted += 1
            ops[op] = ops.get(op, 0) + 1
    return {"accepted": accepted, "ops": ops}


def append_legacy_grounding(
    out: Path,
    manifest: Path,
    ground_target: int,
    salesforce_target: int,
):
    """Add grounding sources without making either source all-or-nothing."""
    rows: list[dict] = []
    writer = ZipShardWriter(out, max_images=750)
    stats: dict[str, object] = {}
    try:
        before = len(rows)
        try:
            stats["groundcua"] = add_groundcua(rows, writer, ground_target)
        except Exception as exc:
            stats["groundcua"] = {
                "status": "partial_error",
                "error": f"{type(exc).__name__}: {exc}",
                "rows_buffered": len(rows) - before,
            }
            print(f"V07_DEV_CU_SOURCE_ERROR groundcua {type(exc).__name__}: {exc}", flush=True)

        before = len(rows)
        try:
            stats["salesforce"] = add_salesforce(rows, writer, salesforce_target)
        except Exception as exc:
            stats["salesforce"] = {
                "status": "partial_error",
                "error": f"{type(exc).__name__}: {exc}",
                "rows_buffered": len(rows) - before,
            }
            print(f"V07_DEV_CU_SOURCE_ERROR salesforce {type(exc).__name__}: {exc}", flush=True)
    finally:
        writer.close()

    converted = _append_legacy_rows(manifest, rows)
    stats.update(converted)
    stats["buffered"] = len(rows)
    stats["image_bytes"] = writer.total_bytes
    return stats


def add_showui(out: Path, manifest: Path, limit: int):
    ds = load_dataset("showlab/ShowUI-desktop", split="train", streaming=True)
    writer = ImageShardWriter(out, prefix="showui_images", max_images=600)
    added = 0
    types: dict[str, int] = {}
    try:
        with manifest.open("a", encoding="utf-8") as fh:
            for row in ds:
                image = row.get("image")
                point = row.get("point")
                instruction = str(row.get("instruction") or "").strip()
                if image is None or not point or len(point) < 2 or not instruction:
                    continue
                x, y = float(point[0]), float(point[1])
                if x > 1.5 or y > 1.5:
                    w, h = image.size
                    x /= max(1, w)
                    y /= max(1, h)
                x = min(1.0, max(0.0, x))
                y = min(1.0, max(0.0, y))
                typ = str(row.get("type") or "click")
                archive, member, _, _ = writer.add(image, f"showui_{added:07d}.jpg")
                rec = {
                    "archive": archive,
                    "image": member,
                    "task": instruction,
                    "operation": "CLICK",
                    "coord": [x, y],
                    "coord2": None,
                    "payload": "",
                    "domain": "showui:" + typ,
                    "source": "showlab/ShowUI-desktop",
                    "episode_id": f"showui:{added}",
                    "step": 0,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                added += 1
                types[typ] = types.get(typ, 0) + 1
                if added % 1000 == 0:
                    print(f"V07_DEV_CU showui_progress={added}/{limit}", flush=True)
                if added >= limit:
                    break
    finally:
        writer.close()
    if added < min(2_000, int(limit * 0.50)):
        raise RuntimeError(f"ShowUI underfilled {added}/{limit}")
    return {"examples": added, "types": types}


def add_click100k(out: Path, manifest: Path, limit: int):
    ds = load_dataset("mlfoundations/Click-100k", split="train", streaming=True)
    writer = ImageShardWriter(out, prefix="click100k_images", max_images=600)
    added = 0
    try:
        with manifest.open("a", encoding="utf-8") as fh:
            for row in ds:
                images = row.get("images")
                image = images[0] if isinstance(images, list) and images else None
                box = row.get("normalized_bbox")
                prompt = str(row.get("easyr1_prompt") or "").strip()
                if image is None or not box or len(box) != 4 or not prompt:
                    continue
                x = (float(box[0]) + float(box[2])) / 2
                y = (float(box[1]) + float(box[3])) / 2
                archive, member, _, _ = writer.add(image, f"click100k_{added:07d}.jpg")
                task = prompt.split("<image>")[-1].strip() or prompt[-900:]
                rec = {
                    "archive": archive,
                    "image": member,
                    "task": task[:1800],
                    "operation": "CLICK",
                    "coord": [min(1, max(0, x)), min(1, max(0, y))],
                    "coord2": None,
                    "payload": "",
                    "domain": "click100k:desktop",
                    "source": "mlfoundations/Click-100k",
                    "episode_id": f"click100k:{added}",
                    "step": 0,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                added += 1
                if added % 1000 == 0:
                    print(f"V07_DEV_CU click100k_progress={added}/{limit}", flush=True)
                if added >= limit:
                    break
    finally:
        writer.close()
    if added < min(1_000, int(limit * 0.40)):
        raise RuntimeError(f"Click-100k underfilled {added}/{limit}")
    return {"examples": added}


def _validate_archives(out: Path, manifest_stats: dict) -> dict:
    missing = []
    empty = []
    for name in manifest_stats.get("archives", {}):
        p = out / name
        if not p.is_file():
            missing.append(name)
        elif p.stat().st_size <= 32:
            empty.append(name)
    if missing or empty:
        raise RuntimeError(
            f"image shard validation failed missing={missing[:8]} empty={empty[:8]}"
        )
    return {
        "archives": len(manifest_stats.get("archives", {})),
        "missing": len(missing),
        "empty": len(empty),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prepared_v07_dev_computer")
    ap.add_argument("--owner", default="owner")
    ap.add_argument("--rexx-trajectories", type=int, default=5000)
    ap.add_argument("--groundcua", type=int, default=8000)
    ap.add_argument("--salesforce", type=int, default=18000)
    ap.add_argument("--showui", type=int, default=7496)
    ap.add_argument("--click100k-max", type=int, default=6000)
    ap.add_argument("--min-rows", type=int, default=30000)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stats: dict[str, object] = {}
    manifest = out / "cu07_manifest.jsonl"

    _safe_stage(
        "rexx",
        lambda: prepare_rexx(out, args.rexx_trajectories),
        stats,
        out,
        required=True,
    )

    _safe_stage(
        "legacy_grounding",
        lambda: append_legacy_grounding(
            out, manifest, args.groundcua, args.salesforce
        ),
        stats,
        out,
        required=False,
    )

    _safe_stage(
        "showui",
        lambda: add_showui(out, manifest, args.showui),
        stats,
        out,
        required=False,
    )

    current = _manifest_stats(manifest)
    needed = max(0, args.min_rows - current["examples"])
    if needed:
        target = min(args.click100k_max, max(1200, needed + 800))
        _safe_stage(
            "click100k_fallback",
            lambda: add_click100k(out, manifest, target),
            stats,
            out,
            required=False,
        )
    else:
        stats["click100k_fallback"] = {
            "status": "skipped",
            "reason": "minimum row target already satisfied",
        }
        _write_progress(out, stats, "click100k_skipped")

    final = _manifest_stats(manifest)
    rows = final["examples"]
    ops = final["ops"]
    if rows < args.min_rows:
        raise RuntimeError(
            f"v0.7 dev computer dataset too small: {rows}/{args.min_rows}; "
            f"parts={json.dumps(stats, ensure_ascii=False)}"
        )
    if ops.get("CLICK", 0) < 20_000:
        raise RuntimeError(f"insufficient CLICK coverage: {ops}")
    if ops.get("KEY", 0) < 100 or ops.get("TYPE", 0) < 10:
        raise RuntimeError(
            "insufficient native keyboard/type diversity from REXX; "
            f"refusing to synthesize fake ComputerUse actions: {ops}"
        )

    archive_check = _validate_archives(out, final)
    summary = {
        "pipeline_version": "v0.7-dev-computer",
        "training_pipeline": "v0.7",
        "stats": {
            "examples": rows,
            "ops": ops,
            "sources": final["sources"],
            "domains": final["domains"],
            "archives": archive_check,
            "parts": stats,
        },
    }
    (out / "sources.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dataset_meta = {
        "title": "FLM v07 DEV Computer Data",
        "id": f"{args.owner}/flm-v07-dev-computer",
        "licenses": [{"name": "other"}],
        "description": "GitHub-CPU prepared FLM v0.7 ComputerUse screenshot/action data.",
    }
    (out / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_progress(out, stats, "complete")
    print("V07_DEV_COMPUTER_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
