#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from flm.tooling.protocol import extract_tool_calls


def natural_text(text: str, *, min_chars: int = 20) -> bool:
    text = str(text or "").strip()
    if len(text) < min_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    weird = sum((ord(ch) < 32 and ch not in "\n\t") or ch == "\ufffd" for ch in text)
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2,}", text)
    return letters / max(1, len(text)) > 0.45 and weird == 0 and len(words) >= 4


def validate(root: Path) -> list[str]:
    errors = []
    suite = json.loads((root / "suite_metrics.json").read_text(encoding="utf-8"))
    if suite.get("pipeline_version") != "v0.7":
        errors.append("wrong pipeline version")
    quality = suite.get("quality_samples") or {}
    main = quality.get("main") or []
    coder = quality.get("coder") or []
    tool = (quality.get("tool_call") or {}).get("output", "")

    if len(main) < 4:
        errors.append("missing main quality samples")
    else:
        for i, item in enumerate(main):
            if not natural_text(item.get("output", "")):
                errors.append(f"main sample {i} is not coherent-looking text: {item.get('output','')[:120]!r}")
        capital = next((x.get("output", "") for x in main if "başkenti" in x.get("prompt", "")), "")
        if "ankara" not in capital.lower():
            errors.append(f"capital QA failed: {capital!r}")
        english = next((x.get("output", "") for x in main if x.get("prompt", "").startswith("Hello")), "")
        if not any(k in english.lower() for k in ("system", "computer", "software", "hardware", "resource", "program")):
            errors.append(f"English OS answer looks off-topic: {english!r}")

    if len(coder) < 2:
        errors.append("missing coder quality samples")
    else:
        for i, item in enumerate(coder):
            out = item.get("output", "")
            if "def " not in out or "return" not in out:
                errors.append(f"coder sample {i} missing Python function structure: {out[:180]!r}")

    try:
        calls = extract_tool_calls(tool)
        if not calls or calls[0].name != "add":
            errors.append(f"tool call did not select add: {tool!r}")
        else:
            args = calls[0].arguments
            if int(args.get("a", -1)) != 27 or int(args.get("b", -1)) != 15:
                errors.append(f"tool arguments incorrect: {args}")
    except Exception as exc:
        errors.append(f"tool call is not parseable: {type(exc).__name__}: {exc}; output={tool!r}")

    cu = (suite.get("models") or {}).get("computer_use") or {}
    if cu:
        if float(cu.get("op_accuracy", 0)) < 0.50:
            errors.append(f"CU held-out op_accuracy too low: {cu.get('op_accuracy')}")
        if float(cu.get("pointer_hit_0p1", 0)) < 0.15:
            errors.append(f"CU held-out pointer hit too low: {cu.get('pointer_hit_0p1')}")
        if not (0 <= float(cu.get("pointer_mean_l2", 99)) < 0.50):
            errors.append(f"CU pointer L2 invalid/high: {cu.get('pointer_mean_l2')}")
    else:
        errors.append("missing ComputerUse metrics")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    root = Path(args.root)
    errors = validate(root)
    if errors:
        print("V07_QUALITY_GATE_FAILED")
        for e in errors:
            print(" -", e)
        return 1
    print("V07_QUALITY_GATE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
