from __future__ import annotations

from dataclasses import dataclass
from collections import deque


@dataclass(frozen=True)
class ActionSchema:
    name:str
    preconditions:frozenset[str]
    add_effects:frozenset[str]
    delete_effects:frozenset[str]=frozenset()
    cost:float=1.0
    payload:dict|None=None

    def applicable(self,state:frozenset[str])->bool:
        return self.preconditions.issubset(state)

    def apply(self,state:frozenset[str])->frozenset[str]:
        if not self.applicable(state):
            raise ValueError(f"action {self.name} preconditions are not satisfied")
        return frozenset((state-self.delete_effects)|self.add_effects)


@dataclass
class PlanResult:
    found:bool
    actions:list[ActionSchema]
    final_state:frozenset[str]
    explored:int


class StatePlanner:
    """Training-free state-space planner with deterministic BFS."""

    def __init__(self,actions:list[ActionSchema],*,max_states:int=10000):
        self.actions=list(actions)
        self.max_states=int(max_states)

    def plan(self,start:set[str]|frozenset[str],goal:set[str]|frozenset[str])->PlanResult:
        start=frozenset(start)
        goal=frozenset(goal)
        if goal.issubset(start):
            return PlanResult(True,[],start,0)
        queue=deque([(start,[])])
        seen={start}
        explored=0
        while queue and explored<self.max_states:
            state,path=queue.popleft()
            explored+=1
            for action in self.actions:
                if not action.applicable(state):
                    continue
                nxt=action.apply(state)
                if nxt in seen:
                    continue
                new_path=path+[action]
                if goal.issubset(nxt):
                    return PlanResult(True,new_path,nxt,explored)
                seen.add(nxt)
                queue.append((nxt,new_path))
        return PlanResult(False,[],start,explored)
