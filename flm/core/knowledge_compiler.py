from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .semantic_compiler import SemanticCompiler
from .semantic_ir import Atom, Program, Rule
from .semantic_memory import SemanticMemory


@dataclass
class IngestStats:
    accepted: int = 0
    rejected: int = 0
    facts_added: int = 0
    rules_added: int = 0


class SemanticKnowledgeCompiler:
    """Compile source data into FLM's training-free semantic memory.

    Raw source wording is never persisted in SemanticMemory. The compiler keeps
    only structured facts/rules. A one-way hash may be returned for provenance,
    but the original sentence is deliberately discarded so the answer composer
    cannot simply echo the training/source text.
    """

    def __init__(self, compiler: SemanticCompiler | None = None):
        self.compiler=compiler or SemanticCompiler()

    @staticmethod
    def _structured_record(obj: dict) -> Program | None:
        if any(k in obj for k in ("facts","rules")):
            return Program.from_dict({
                "facts":obj.get("facts") or [],
                "rules":obj.get("rules") or [],
            })
        subject=str(obj.get("subject") or "").strip()
        predicate=str(obj.get("predicate") or "").strip()
        value=obj.get("object",obj.get("value"))
        if subject and predicate and value is not None:
            args=value if isinstance(value,list) else [value]
            return Program(facts=[Atom(
                predicate,
                tuple([subject]+[str(x) for x in args]),
            )])
        return None

    def compile_record(self, record: str | dict) -> tuple[Program,str]:
        raw_text=""
        if isinstance(record,str):
            raw_text=record
            program=self.compiler.compile_any(record)
        elif isinstance(record,dict):
            program=self._structured_record(record)
            if program is None:
                raw_text=str(record.get("text") or record.get("sentence") or "").strip()
                if not raw_text:
                    raise ValueError("record has neither structured facts/rules nor compilable text")
                program=self.compiler.compile_any(raw_text)
        else:
            raise TypeError("knowledge record must be text or dict")

        # Never carry source metadata or raw wording into long-term memory.
        clean=Program(facts=list(program.facts),rules=list(program.rules))
        fingerprint=hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else ""
        return clean,fingerprint

    def ingest(self, memory: SemanticMemory, records: Iterable[str | dict]) -> IngestStats:
        stats=IngestStats()
        before=memory.stats()
        for record in records:
            try:
                program,_=self.compile_record(record)
                memory.ingest(program)
                stats.accepted+=1
            except (ValueError,TypeError,KeyError):
                stats.rejected+=1
        after=memory.stats()
        stats.facts_added=after.facts-before.facts
        stats.rules_added=after.rules-before.rules
        return stats

    def ingest_jsonl(self, memory: SemanticMemory, path: str | Path) -> IngestStats:
        p=Path(path)
        records=[]
        with p.open("r",encoding="utf-8") as fh:
            for line in fh:
                line=line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append(line)
        return self.ingest(memory,records)
