#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from flm.inference_v07 import V07TextAgent
from flm.tooling.runtime import ToolRegistry


def main() -> int:
    ap = argparse.ArgumentParser(description="Run FLM v0.7 Main/Coder chat and tool calls")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system")
    ap.add_argument("--device", choices=("cpu", "cuda"), default=None)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.20)
    ap.add_argument(
        "--demo-add-tool",
        action="store_true",
        help="Register a safe add(a,b) tool and run the full tool-call loop",
    )
    args = ap.parse_args()

    agent = V07TextAgent(
        args.checkpoint,
        args.tokenizer,
        device=args.device,
        temperature=args.temperature,
        max_new=args.max_new,
    )
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    if not args.demo_add_tool:
        print(agent.generate_history(messages))
        return 0

    registry = ToolRegistry()
    registry.register(
        "add",
        lambda a, b: int(a) + int(b),
        description="Add two integers.",
        schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
    output, history = agent.run_tools(messages, registry)
    print(output)
    print(json.dumps(history, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
