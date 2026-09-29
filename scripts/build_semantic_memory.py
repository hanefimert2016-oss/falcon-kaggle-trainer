#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flm.core import FLMCore, SemanticKnowledgeCompiler


def main()->int:
    ap=argparse.ArgumentParser(
        description="Compile knowledge into source-hidden FLM Semantic Memory. No gradient/model training."
    )
    ap.add_argument("--input",type=Path,required=True,help="JSONL or canonical text")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--merge",type=Path)
    args=ap.parse_args()

    core=FLMCore.load(args.merge) if args.merge else FLMCore()
    compiler=SemanticKnowledgeCompiler()
    stats=compiler.ingest_jsonl(core.memory,args.input)
    core.save(args.out)
    mem=core.memory.stats()
    result={
        "format":"flm-semantic-memory-build-v2",
        "accepted":stats.accepted,
        "rejected":stats.rejected,
        "facts_added":stats.facts_added,
        "rules_added":stats.rules_added,
        "facts":mem.facts,
        "rules":mem.rules,
        "predicates":mem.predicates,
        "output":str(args.out),
        "bytes":args.out.stat().st_size,
        "training_steps":0,
        "raw_source_persisted":False,
        "neural_model_required":False,
    }
    print("FLM_SEMANTIC_MEMORY_READY="+json.dumps(result,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
