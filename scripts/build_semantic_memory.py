#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flm.core import FLMCore, Program
from flm.core.semantic_compiler import SemanticCompiler


def main()->int:
    ap=argparse.ArgumentParser(
        description="Compile structured/canonical knowledge directly into FLM Semantic Memory. No training."
    )
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ir-jsonl",type=Path)
    src.add_argument("--canonical-text",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--merge",type=Path)
    args=ap.parse_args()

    core=FLMCore.load(args.merge) if args.merge else FLMCore()
    compiler=SemanticCompiler()
    rows=accepted=rejected=0

    source=args.ir_jsonl or args.canonical_text
    with source.open("r",encoding="utf-8",errors="ignore") as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            rows+=1
            try:
                if args.ir_jsonl:
                    obj=json.loads(line)
                    if "program" in obj:
                        obj=obj["program"]
                    program=Program.from_dict(obj)
                else:
                    program=compiler.compile_canonical(line)
                core.ingest(program)
                accepted+=1
            except Exception:
                rejected+=1

    core.save(args.out)
    stats=core.memory.stats()
    result={
        "format":"flm-semantic-memory-build-v1",
        "rows":rows,
        "accepted":accepted,
        "rejected":rejected,
        "facts":stats.facts,
        "rules":stats.rules,
        "predicates":stats.predicates,
        "output":str(args.out),
        "bytes":args.out.stat().st_size,
        "training_steps":0,
    }
    print("FLM_SEMANTIC_MEMORY_READY="+json.dumps(result,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
