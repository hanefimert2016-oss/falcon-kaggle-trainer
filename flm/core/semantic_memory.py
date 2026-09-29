from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from .semantic_ir import Atom, Rule, Program


@dataclass
class MemoryStats:
    facts: int
    rules: int
    predicates: int


class SemanticMemory:
    """Training-free symbolic memory used directly by FLM Core.

    This is model state, not a vector database: facts/rules are indexed by exact
    predicates and consumed by the reasoning VM rather than retrieved as text.
    """

    def __init__(self):
        self._facts: set[Atom] = set()
        self._facts_by_predicate: dict[str, set[Atom]] = defaultdict(set)
        self._facts_by_subject: dict[str, set[Atom]] = defaultdict(set)
        self._rules: list[Rule] = []
        self._rules_by_head: dict[str, list[Rule]] = defaultdict(list)

    def add_fact(self, atom: Atom) -> bool:
        if atom in self._facts:
            return False
        self._facts.add(atom)
        self._facts_by_predicate[atom.predicate].add(atom)
        if atom.args:
            self._facts_by_subject[str(atom.args[0])].add(atom)
        return True

    def add_rule(self, rule: Rule) -> None:
        if rule not in self._rules:
            self._rules.append(rule)
            self._rules_by_head[rule.conclusion.predicate].append(rule)

    def ingest(self, program: Program) -> None:
        for fact in program.facts:
            self.add_fact(fact)
        for rule in program.rules:
            self.add_rule(rule)

    def facts(self, predicate: str | None = None):
        if predicate is None:
            return tuple(sorted(self._facts))
        return tuple(sorted(self._facts_by_predicate.get(predicate, ())))

    def facts_about(self, subject: str):
        return tuple(sorted(self._facts_by_subject.get(str(subject), ())))

    def rules(self, head_predicate: str | None = None):
        if head_predicate is None:
            return tuple(self._rules)
        return tuple(self._rules_by_head.get(head_predicate, ()))

    def stats(self) -> MemoryStats:
        predicates = set(self._facts_by_predicate) | set(self._rules_by_head)
        return MemoryStats(len(self._facts), len(self._rules), len(predicates))

    def save(self, path: str | Path) -> None:
        p=Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj={
            "format":"flm-semantic-memory-v1",
            "facts":[x.to_dict() for x in sorted(self._facts)],
            "rules":[x.to_dict() for x in self._rules],
        }
        p.write_text(json.dumps(obj,ensure_ascii=False,separators=(",",":")),encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticMemory":
        obj=json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("format")!="flm-semantic-memory-v1":
            raise RuntimeError(f"unsupported semantic memory format: {obj.get('format')!r}")
        mem=cls()
        mem.ingest(Program.from_dict(obj))
        return mem
