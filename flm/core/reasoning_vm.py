from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .semantic_ir import Atom, Rule, is_variable
from .semantic_memory import SemanticMemory


def unify(pattern: Atom, fact: Atom, env: dict[str, str] | None = None):
    if pattern.predicate != fact.predicate or len(pattern.args) != len(fact.args):
        return None
    out=dict(env or {})
    for p, f in zip(pattern.args, fact.args):
        if is_variable(p):
            current=out.get(p)
            if current is not None and current != f:
                return None
            out[p]=f
        elif p != f:
            return None
    return out


@dataclass
class ReasoningResult:
    answer: bool
    query: Atom
    proof: list[dict]
    rounds: int
    derived_facts: int


class ReasoningVM:
    """Deterministic rule engine: no gradient training and no RAG."""

    def __init__(self, memory: SemanticMemory, *, max_rounds: int = 32):
        self.memory=memory
        self.max_rounds=int(max_rounds)

    def _bindings(self, premises: tuple[Atom, ...], facts: set[Atom]):
        envs=[{}]
        for premise in premises:
            candidates=[x for x in facts if x.predicate==premise.predicate]
            next_envs=[]
            for env in envs:
                bound=premise.substitute(env)
                for fact in candidates:
                    merged=unify(bound,fact,env)
                    if merged is not None:
                        next_envs.append(merged)
            envs=next_envs
            if not envs:
                break
        return envs

    def infer(self):
        facts=set(self.memory.facts())
        provenance={fact:{"kind":"fact"} for fact in facts}
        derived=0
        rounds=0
        for rounds in range(1,self.max_rounds+1):
            added=[]
            for rule in self.memory.rules():
                for env in self._bindings(rule.premises,facts):
                    conclusion=rule.conclusion.substitute(env)
                    if any(is_variable(x) for x in conclusion.args):
                        continue
                    if conclusion not in facts:
                        premises=[x.substitute(env) for x in rule.premises]
                        added.append((conclusion,rule,premises))
            if not added:
                break
            for conclusion,rule,premises in added:
                if conclusion in facts:
                    continue
                facts.add(conclusion)
                provenance[conclusion]={
                    "kind":"rule",
                    "rule":rule.name or rule.conclusion.predicate,
                    "premises":premises,
                }
                derived+=1
        return facts,provenance,rounds,derived

    def _proof(self, atom: Atom, provenance: dict[Atom,dict], seen=None):
        seen=set(seen or ())
        if atom in seen:
            return [{"atom":atom.to_dict(),"kind":"cycle"}]
        seen.add(atom)
        info=provenance.get(atom)
        if info is None:
            return []
        node={"atom":atom.to_dict(),"kind":info["kind"]}
        if info["kind"]=="rule":
            node["rule"]=info["rule"]
            node["premises"]=[
                self._proof(x,provenance,seen)
                for x in info["premises"]
            ]
        return [node]

    def prove(self, query: Atom) -> ReasoningResult:
        facts,prov,rounds,derived=self.infer()
        ok=query in facts
        return ReasoningResult(
            answer=ok,
            query=query,
            proof=self._proof(query,prov) if ok else [],
            rounds=rounds,
            derived_facts=derived,
        )
