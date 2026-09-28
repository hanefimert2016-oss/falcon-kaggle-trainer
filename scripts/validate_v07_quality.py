#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flm.quality_v07 import validate_suite


def validate(root: Path) -> list[str]:
    suite = json.loads((root / "suite_metrics.json").read_text(encoding="utf-8"))
    return validate_suite(suite)


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
