from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

from .semantic_ir import Atom, Rule, Program, is_variable


def _alias_key(value: str) -> str:
    text=unicodedata.normalize("NFKC",str(value or "")).casefold().strip()
    return re.sub(r"\s+"," ",text)


@dataclass
class MemoryStats:
    facts: int
    rules: int
    predicates: int
    entity_aliases: int = 0
    entity_labels: int = 0
    predicate_labels: int = 0


class SemanticMemory:
    """Training-free symbolic memory used directly by FLM Core.

    Facts/rules are model state, not retrieved source chunks. Raw source prose is
    not required. Optional aliases and display labels map human names to stable
    canonical IDs without gradient training.
    """

    def __init__(self):
        self._facts: set[Atom] = set()
        self._facts_by_predicate: dict[str, set[Atom]] = defaultdict(set)
        self._facts_by_subject: dict[str, set[Atom]] = defaultdict(set)
        self._rules: list[Rule] = []
        self._rules_by_head: dict[str, list[Rule]] = defaultdict(list)
        self._entity_aliases: dict[str,str] = {}
        self._entity_labels: dict[str,str] = {}
        self._predicate_labels: dict[str,str] = {}

    def add_entity_alias(
        self,
        alias: str,
        canonical: str,
        *,
        display: str | None = None,
    ) -> None:
        canonical=str(canonical).strip()
        alias=str(alias).strip()
        if not canonical or not alias:
            return
        self._entity_aliases[_alias_key(alias)]=canonical
        self._entity_aliases[_alias_key(canonical)]=canonical
        if display:
            self._entity_labels.setdefault(canonical,str(display).strip())
            self._entity_aliases[_alias_key(display)]=canonical

    def set_entity_label(self, canonical: str, label: str) -> None:
        canonical=str(canonical).strip()
        label=str(label).strip()
        if not canonical or not label:
            return
        self._entity_labels[canonical]=label
        self.add_entity_alias(label,canonical,display=label)

    def set_predicate_label(self, predicate: str, label: str) -> None:
        predicate=str(predicate).strip()
        label=str(label).strip()
        if predicate and label:
            self._predicate_labels[predicate]=label

    def resolve_entity(self, value: str) -> str:
        raw=str(value)
        return self._entity_aliases.get(_alias_key(raw),raw)

    def display_entity(self, value: str) -> str:
        canonical=self.resolve_entity(value)
        return self._entity_labels.get(canonical,canonical.replace("_"," "))

    def predicate_label(self, predicate: str) -> str:
        return self._predicate_labels.get(str(predicate),str(predicate).replace("_"," "))

    def resolve_atom(self, atom: Atom) -> Atom:
        return Atom(
            atom.predicate,
            tuple(
                x if is_variable(x) else self.resolve_entity(x)
                for x in atom.args
            ),
        )

    def add_fact(self, atom: Atom) -> bool:
        atom=self.resolve_atom(atom)
        if atom in self._facts:
            return False
        self._facts.add(atom)
        self._facts_by_predicate[atom.predicate].add(atom)
        if atom.args:
            self._facts_by_subject[str(atom.args[0])].add(atom)
        return True

    def add_rule(self, rule: Rule) -> None:
        resolved=Rule(
            tuple(self.resolve_atom(x) for x in rule.premises),
            self.resolve_atom(rule.conclusion),
            rule.name,
        )
        if resolved not in self._rules:
            self._rules.append(resolved)
            self._rules_by_head[resolved.conclusion.predicate].append(resolved)

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
        subject=self.resolve_entity(subject)
        return tuple(sorted(self._facts_by_subject.get(str(subject), ())))

    def rules(self, head_predicate: str | None = None):
        if head_predicate is None:
            return tuple(self._rules)
        return tuple(self._rules_by_head.get(head_predicate, ()))

    def stats(self) -> MemoryStats:
        predicates = set(self._facts_by_predicate) | set(self._rules_by_head)
        return MemoryStats(
            len(self._facts),len(self._rules),len(predicates),
            len(self._entity_aliases),len(self._entity_labels),len(self._predicate_labels),
        )

    def save(self, path: str | Path) -> None:
        p=Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj={
            "format":"flm-semantic-memory-v2",
            "facts":[x.to_dict() for x in sorted(self._facts)],
            "rules":[x.to_dict() for x in self._rules],
            "entity_aliases":self._entity_aliases,
            "entity_labels":self._entity_labels,
            "predicate_labels":self._predicate_labels,
        }
        p.write_text(json.dumps(obj,ensure_ascii=False,separators=(",",":")),encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticMemory":
        obj=json.loads(Path(path).read_text(encoding="utf-8"))
        fmt=obj.get("format")
        if fmt not in {"flm-semantic-memory-v1","flm-semantic-memory-v2"}:
            raise RuntimeError(f"unsupported semantic memory format: {fmt!r}")
        mem=cls()
        if fmt=="flm-semantic-memory-v2":
            mem._entity_aliases={
                str(k):str(v) for k,v in (obj.get("entity_aliases") or {}).items()
            }
            mem._entity_labels={
                str(k):str(v) for k,v in (obj.get("entity_labels") or {}).items()
            }
            mem._predicate_labels={
                str(k):str(v) for k,v in (obj.get("predicate_labels") or {}).items()
            }
        mem.ingest(Program.from_dict(obj))
        return mem
