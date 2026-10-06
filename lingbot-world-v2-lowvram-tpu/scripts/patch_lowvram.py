#!/usr/bin/env python3
"""Patch an upstream LingBot-World-V2 clone for arbitrary max-area and 8GB presets.

The patch is intentionally small and reversible. It does not replace model math.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find patch anchor: {label}")
    return text.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("upstream", type=Path)
    args = ap.parse_args()
    gen = args.upstream / "generate.py"
    s = gen.read_text()

    s = replace_once(
        s,
        'parser.add_argument(\n        "--frame_num",',
        'parser.add_argument(\n        "--max_area_override",\n        type=int,\n        default=None,\n        help="Override SIZE_CONFIGS area. Useful for low-VRAM 8GB presets."\n    )\n    parser.add_argument(\n        "--frame_num",',
        "max_area arg",
    )
    s = replace_once(
        s,
        '    # Size check\n    assert args.size in SUPPORTED_SIZES[\n        args.\n        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {\', \'.join(SUPPORTED_SIZES[args.task])}"',
        '    # Size check. A max-area override intentionally bypasses the fixed preset list;\n    # Wan derives the final H/W from input aspect ratio and this area.\n    if args.max_area_override is None:\n        assert args.size in SUPPORTED_SIZES[\n            args.task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {\', \'.join(SUPPORTED_SIZES[args.task])}"',
        "validation override",
    )
    s = replace_once(
        s,
        '        max_area=MAX_AREA_CONFIGS[args.size],',
        '        max_area=(args.max_area_override if args.max_area_override is not None else MAX_AREA_CONFIGS[args.size]),',
        "generation max area",
    )
    gen.write_text(s)
    print(f"Patched {gen}")


if __name__ == "__main__":
    main()
