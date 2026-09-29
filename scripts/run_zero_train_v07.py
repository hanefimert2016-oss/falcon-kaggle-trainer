#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from flm.core import FLMCore, SemanticKnowledgeCompiler
from flm.core_agent import TrainingFreeAgent


def main()->int:
    ap=argparse.ArgumentParser(description="Run FLM v0.7 strict zero-train Core")
    ap.add_argument("--memory",type=Path)
    ap.add_argument("--ingest",type=Path,action="append",default=[])
    ap.add_argument("--save-memory",type=Path)
    ap.add_argument("--prompt")
    args=ap.parse_args()

    core=FLMCore.load(args.memory) if args.memory else FLMCore()
    compiler=SemanticKnowledgeCompiler()
    for source in args.ingest:
        stats=compiler.ingest_jsonl(core.memory,source)
        print(
            f"INGEST {source}: accepted={stats.accepted} rejected={stats.rejected} "
            f"facts_added={stats.facts_added} rules_added={stats.rules_added}"
        )

    agent=TrainingFreeAgent(core)
    if args.prompt is not None:
        print(agent.ask(args.prompt))
    else:
        print("FLM v0.7 strict zero-train. Çıkmak için Ctrl-D/Ctrl-C.")
        while True:
            try:
                prompt=input("> ").strip()
            except (EOFError,KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            try:
                print(agent.ask(prompt))
            except Exception as exc:
                print(f"ANLAŞILMADI: {type(exc).__name__}: {exc}")

    if args.save_memory:
        core.save(args.save_memory)
        print(f"MEMORY_SAVED {args.save_memory}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
