from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .semantic_ir import Program, Query
from .semantic_memory import SemanticMemory
from .reasoning_vm import ReasoningVM, ReasoningResult


@dataclass
class CoreResponse:
    results: list[ReasoningResult]
    memory_facts: int
    memory_rules: int


class FLMCore:
    """Training-free main intelligence core.

    Neural models may translate language <-> Program, but reasoning and stored
    knowledge execution are deterministic here.
    """

    def __init__(self, memory: SemanticMemory | None = None):
        self.memory=memory or SemanticMemory()

    def ingest(self, program: Program) -> None:
        self.memory.ingest(program)

    def execute(self, program: Program) -> CoreResponse:
        self.memory.ingest(Program(facts=program.facts,rules=program.rules))
        vm=ReasoningVM(self.memory)
        results=[vm.prove(q.atom) for q in program.queries]
        stats=self.memory.stats()
        return CoreResponse(results,stats.facts,stats.rules)

    def save(self, path: str | Path) -> None:
        self.memory.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "FLMCore":
        return cls(SemanticMemory.load(path))
