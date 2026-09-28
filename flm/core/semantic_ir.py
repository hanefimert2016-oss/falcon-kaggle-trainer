from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import re


_VAR = re.compile(r"^\?[A-Za-z_][A-Za-z0-9_]*$")


def is_variable(value: str) -> bool:
    return bool(_VAR.match(str(value)))


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    args: tuple[str, ...]

    def __post_init__(self):
        if not self.predicate or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.predicate):
            raise ValueError(f"invalid predicate: {self.predicate!r}")
        if not self.args:
            raise ValueError("atom requires at least one argument")

    def substitute(self, env: dict[str, str]) -> "Atom":
        return Atom(self.predicate, tuple(env.get(x, x) for x in self.args))

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "args": list(self.args)}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Atom":
        return cls(str(obj["predicate"]), tuple(str(x) for x in obj["args"]))


@dataclass(frozen=True)
class Rule:
    premises: tuple[Atom, ...]
    conclusion: Atom
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "premises": [x.to_dict() for x in self.premises],
            "conclusion": self.conclusion.to_dict(),
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Rule":
        return cls(
            tuple(Atom.from_dict(x) for x in obj.get("premises", [])),
            Atom.from_dict(obj["conclusion"]),
            str(obj.get("name") or ""),
        )


@dataclass(frozen=True)
class Query:
    atom: Atom


@dataclass
class Program:
    facts: list[Atom] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [x.to_dict() for x in self.facts],
            "rules": [x.to_dict() for x in self.rules],
            "queries": [{"atom": q.atom.to_dict()} for q in self.queries],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Program":
        return cls(
            facts=[Atom.from_dict(x) for x in obj.get("facts", [])],
            rules=[Rule.from_dict(x) for x in obj.get("rules", [])],
            queries=[Query(Atom.from_dict(x["atom"])) for x in obj.get("queries", [])],
            metadata=dict(obj.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Program":
        return cls.from_dict(json.loads(raw))
