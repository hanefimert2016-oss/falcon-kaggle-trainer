#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import time

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

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


UNIGUI_REPO = "UI-MOPD/Uni-GUI-Desktop-1"
UNIGUI_CAPS = {
    "CLICK": 2000,
    "DOUBLE_CLICK": 900,
    "RIGHT_CLICK": 900,
    "DRAG": 1600,
    "SCROLL": 2400,
    "TYPE": 2400,
    "KEY": 2400,
    "MOVE": 900,
    "WAIT": 900,
    "DONE": 2200,
}


def _unigui_op(step: dict) -> str | None:
    plan=step.get("plan") or {}
    args=plan.get("arguments") if isinstance(plan,dict) else {}
    if not isinstance(args,dict):
        return None
    name=str(args.get("action") or "").lower()
    return {
        "left_click":"CLICK",
        "double_click":"DOUBLE_CLICK",
        "right_click":"RIGHT_CLICK",
        "left_click_drag":"DRAG",
        "scroll":"SCROLL",
        "type":"TYPE",
        "key":"KEY",
        "mouse_move":"MOVE",
        "wait":"WAIT",
        "terminate":"DONE",
    }.get(name)


def _unigui_plan_fallback(step: dict, op: str) -> dict | None:
    plan=step.get("plan") or {}
    args=plan.get("arguments") if isinstance(plan,dict) else {}
    if not isinstance(args,dict):
        return None

    def norm(v):
        if not isinstance(v,(list,tuple)) or len(v)<2:
            return None
        try:
            x,y=float(v[0]),float(v[1])
        except Exception:
            return None
        if abs(x)>1.5 or abs(y)>1.5:
            x/=999.0
            y/=999.0
        return [min(1.0,max(0.0,x)),min(1.0,max(0.0,y))]

    if op in {"CLICK","DOUBLE_CLICK","RIGHT_CLICK","MOVE"}:
        xy=norm(args.get("coordinate"))
        return {"op":op,"coord":xy,"coord2":None,"payload":""} if xy else None
    if op=="DRAG":
        a=norm(args.get("start_coordinate"))
        b=norm(args.get("coordinate"))
        return {"op":op,"coord":a,"coord2":b,"payload":""} if a and b else None
    if op=="TYPE":
        text=args.get("text",args.get("content",args.get("value","")))
        return {"op":op,"coord":None,"coord2":None,"payload":str(text)} if str(text) else None
    if op=="KEY":
        keys=args.get("keys",args.get("key",args.get("hotkey","")))
        if isinstance(keys,(list,tuple)):
            keys="+".join(map(str,keys))
        return {"op":op,"coord":None,"coord2":None,"payload":str(keys)} if str(keys) else None
    if op=="SCROLL":
        dx=args.get("scroll_x",0)
        dy=args.get("scroll_y",args.get("amount",args.get("delta",0)))
        direction=str(args.get("direction") or args.get("scroll_direction") or "").lower()
        try:
            dx=int(dx or 0); dy=int(dy or 0)
        except Exception:
            dx=0; dy=0
        if direction in {"up","north"} and dy==0: dy=3
        if direction in {"down","south"} and dy==0: dy=-3
        if direction in {"left","west"} and dx==0: dx=-3
        if direction in {"right","east"} and dx==0: dx=3
        return {"op":op,"coord":None,"coord2":None,"payload":f"{dx},{dy}"}
    if op=="WAIT":
        seconds=args.get("seconds",args.get("duration",1))
        return {"op":op,"coord":None,"coord2":None,"payload":str(seconds)}
    if op=="DONE":
        status=str(args.get("status") or "success")
        return {"op":op,"coord":None,"coord2":None,"payload":status}
    return None


def add_unigui_balanced(out: Path, manifest: Path, caps: dict[str,int] | None = None):
    caps=dict(caps or UNIGUI_CAPS)
    meta_dir=out/"_unigui_meta"
    if meta_dir.exists():
        shutil.rmtree(meta_dir)
    snapshot_download(
        repo_id=UNIGUI_REPO,
        repo_type="dataset",
        allow_patterns=["*/task.json"],
        local_dir=meta_dir,
    )
    task_files=sorted(meta_dir.glob("*/task.json"))
    writer=ImageShardWriter(out,prefix="unigui_images",max_images=500)
    counts={k:0 for k in caps}
    trajectories=0
    accepted=0
    try:
        with manifest.open("a",encoding="utf-8") as fh:
            for task_file in task_files:
                try:
                    task=json.loads(task_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if task.get("is_delete") or task.get("is_mock") or not task.get("task_completed",True):
                    continue
                episode=str(task.get("episode_id") or task_file.parent.name)
                query=str(task.get("query") or "").strip()
                app=str(task.get("app") or "desktop")
                if not query:
                    continue
                used=False
                for step in task.get("data") or []:
                    if not isinstance(step,dict) or step.get("is_delete") or step.get("is_use") is False:
                        continue
                    op=_unigui_op(step)
                    if op not in caps or counts[op]>=caps[op]:
                        continue
                    screenshot=str(step.get("screenshot") or "")
                    if not screenshot:
                        continue
                    remote=f"{task_file.parent.name}/{screenshot}"
                    try:
                        image_path=hf_hub_download(
                            repo_id=UNIGUI_REPO,
                            repo_type="dataset",
                            filename=remote,
                        )
                        with Image.open(image_path) as image:
                            width,height=image.size
                            parsed=parse_pyautogui_action(step.get("code") or "",width,height)
                            if parsed is None or parsed.get("op")!=op:
                                parsed=_unigui_plan_fallback(step,op)
                            if parsed is None:
                                continue
                            archive,member,_,_=writer.add(
                                image,
                                f"unigui_{episode}_{int(step.get('step') or accepted):04d}.jpg".replace("/","_"),
                            )
                    except Exception as exc:
                        print(f"V07_DEV_CU unigui_step_error {episode} {op} {type(exc).__name__}: {exc}",flush=True)
                        continue
                    rec={
                        "archive":archive,
                        "image":member,
                        "task":query,
                        "operation":op,
                        "coord":parsed.get("coord"),
                        "coord2":parsed.get("coord2"),
                        "payload":parsed.get("payload",""),
                        "domain":"unigui:"+app,
                        "source":UNIGUI_REPO,
                        "episode_id":"unigui:"+episode,
                        "step":int(step.get("step") or 0),
                    }
                    fh.write(json.dumps(rec,ensure_ascii=False)+"\n")
                    counts[op]+=1
                    accepted+=1
                    used=True
                if used:
                    trajectories+=1
                if all(counts[k]>=v for k,v in caps.items()):
                    break
    finally:
        writer.close()
        shutil.rmtree(meta_dir,ignore_errors=True)
    return {
        "examples":accepted,
        "trajectories":trajectories,
        "ops":counts,
        "caps":caps,
        "source":UNIGUI_REPO,
    }


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
    ap.add_argument("--unigui", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-rows", type=int, default=40000)
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

    if args.unigui:
        _safe_stage(
            "unigui_balanced",
            lambda: add_unigui_balanced(out, manifest),
            stats,
            out,
            required=True,
        )
    else:
        stats["unigui_balanced"]={"status":"disabled"}

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
    minimum_ops = {
        "CLICK": 20_000,
        "KEY": 1_000,
        "TYPE": 1_000,
        "SCROLL": 750,
        "DRAG": 300,
        "RIGHT_CLICK": 200,
        "DOUBLE_CLICK": 200,
        "DONE": 500,
    }
    short={op:(ops.get(op,0),minimum) for op,minimum in minimum_ops.items() if ops.get(op,0)<minimum}
    if short:
        raise RuntimeError(
            "insufficient real multi-action ComputerUse coverage; "
            f"short={short} all_ops={ops}"
        )

    archive_check = _validate_archives(out, final)
    summary = {
        "pipeline_version": "v0.7-dev-computer",
        "data_revision": 2,
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
        "description": "FLM v0.7 r2 ComputerUse data with real balanced multi-action desktop trajectories.",
    }
    (out / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_progress(out, stats, "complete")
    print("V07_DEV_COMPUTER_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
