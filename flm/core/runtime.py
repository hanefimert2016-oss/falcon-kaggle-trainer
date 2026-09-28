from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .semantic_ir import Program, Query
from .semantic_memory import SemanticMemory
from .reasoning_vm import ReasoningVM, ReasoningResult
from .planner import ActionSchema, StatePlanner, PlanResult
from .solvers import ArithmeticSolver
from .verifier import Verifier
from .code_engine import CodeEngine
from .ui_engine import UIPlanner, UIGraph, UIElement


@dataclass
class CoreResponse:
    results: list[ReasoningResult]
    operation_results: list[dict]
    memory_facts: int
    memory_rules: int


class FLMCore:
    """Training-free main intelligence core.

    Neural models may translate language <-> Program, but reasoning and stored
    knowledge execution are deterministic here.
    """

    def __init__(self, memory: SemanticMemory | None = None):
        self.memory=memory or SemanticMemory()
        self.arithmetic=ArithmeticSolver()
        self.verifier=Verifier()
        self.code=CodeEngine()
        self.ui=UIPlanner()

    def ingest(self, program: Program) -> None:
        self.memory.ingest(program)

    def _execute_operation(self, op) -> dict:
        kind=op.kind
        args=dict(op.args)
        if kind=="ARITHMETIC":
            expression=str(args["expression"])
            return {"kind":kind,"ok":True,"value":self.solve_arithmetic(expression)}

        if kind=="ANALYZE_CODE":
            language=str(args.get("language") or "python")
            source=str(args.get("source") or "")
            graph=self.code.analyze_text(source,language)
            return {
                "kind":kind,
                "ok":True,
                "language":graph.language,
                "symbols":[x.__dict__ for x in graph.symbols],
                "calls":[list(x) for x in sorted(graph.calls)],
                "imports":sorted(graph.imports),
                "diagnostics":list(graph.diagnostics),
            }

        if kind=="STATE_PLAN":
            raw_actions=args.get("actions") or []
            actions=[
                ActionSchema(
                    str(x["name"]),
                    frozenset(map(str,x.get("preconditions") or [])),
                    frozenset(map(str,x.get("add_effects") or [])),
                    frozenset(map(str,x.get("delete_effects") or [])),
                    float(x.get("cost",1.0)),
                    dict(x.get("payload") or {}) or None,
                )
                for x in raw_actions
            ]
            result=self.plan(
                set(map(str,args.get("start") or [])),
                set(map(str,args.get("goal") or [])),
                actions,
            )
            return {
                "kind":kind,
                "ok":result.found,
                "actions":[x.name for x in result.actions],
                "final_state":sorted(result.final_state),
                "explored":result.explored,
            }

        if kind=="UI_PLAN":
            elements=[
                UIElement(
                    id=str(x["id"]),
                    role=str(x.get("role") or ""),
                    name=str(x.get("name") or ""),
                    value=str(x.get("value") or ""),
                    enabled=bool(x.get("enabled",True)),
                    visible=bool(x.get("visible",True)),
                    x=float(x["x"]) if x.get("x") is not None else None,
                    y=float(x["y"]) if x.get("y") is not None else None,
                )
                for x in (args.get("elements") or [])
            ]
            graph=UIGraph(str(args.get("app") or "desktop"),elements)
            actions=self.ui.plan(str(args.get("goal") or ""),graph)
            return {
                "kind":kind,
                "ok":bool(actions),
                "actions":[x.__dict__ for x in actions],
            }

        if kind=="VERIFY":
            checks=args.get("checks") or []
            evaluated=[]
            ok=True
            for item in checks:
                actual=item.get("actual")
                expected=item.get("expected")
                passed=actual==expected
                evaluated.append({
                    "name":str(item.get("name") or "check"),
                    "ok":passed,
                    "actual":actual,
                    "expected":expected,
                })
                ok=ok and passed
            return {"kind":kind,"ok":ok,"checks":evaluated}

        raise ValueError(f"unsupported FLM Core operation: {kind}")


    def execute(self, program: Program) -> CoreResponse:
        self.memory.ingest(Program(facts=program.facts,rules=program.rules))
        vm=ReasoningVM(self.memory)
        results=[vm.prove(q.atom) for q in program.queries]
        operation_results=[]
        for op in program.operations:
            try:
                operation_results.append(self._execute_operation(op))
            except Exception as exc:
                operation_results.append({
                    "kind":op.kind,
                    "ok":False,
                    "error":type(exc).__name__,
                    "message":str(exc),
                })
        stats=self.memory.stats()
        return CoreResponse(results,operation_results,stats.facts,stats.rules)

    def solve_arithmetic(self, expression: str):
        return self.arithmetic.solve(expression)

    def plan(self, start, goal, actions: list[ActionSchema]) -> PlanResult:
        return StatePlanner(actions).plan(start, goal)

    def save(self, path: str | Path) -> None:
        self.memory.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "FLMCore":
        return cls(SemanticMemory.load(path))
