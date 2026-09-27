#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from flm.computer_use.agent import build_desktop_agent


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run FLM ComputerUse v0.7 in a closed screenshot -> action loop."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--device", choices=("cpu", "cuda"), default=None)
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually send mouse/keyboard events. Without this flag actions are dry-run only.",
    )
    args = ap.parse_args()
    agent = build_desktop_agent(
        args.checkpoint,
        args.tokenizer,
        execute=args.execute,
        monitor=args.monitor,
        max_steps=args.max_steps,
        device=args.device,
    )
    history = agent.run(args.task)
    print(json.dumps(history, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
