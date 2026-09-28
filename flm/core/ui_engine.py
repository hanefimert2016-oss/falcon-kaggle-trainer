from __future__ import annotations

from dataclasses import dataclass,field
import re


@dataclass(frozen=True)
class UIElement:
    id:str
    role:str
    name:str=""
    value:str=""
    enabled:bool=True
    visible:bool=True
    x:float|None=None
    y:float|None=None


@dataclass
class UIGraph:
    app:str
    elements:list[UIElement]=field(default_factory=list)

    def find(self,*terms:str,roles:tuple[str,...]=()):
        needles=[x.casefold() for x in terms if x]
        out=[]
        for el in self.elements:
            if not el.visible or not el.enabled:
                continue
            if roles and el.role.casefold() not in {x.casefold() for x in roles}:
                continue
            hay=f"{el.name} {el.value} {el.role}".casefold()
            if all(n in hay for n in needles):
                out.append(el)
        return out


@dataclass(frozen=True)
class UIAction:
    op:str
    target_id:str|None=None
    payload:str=""


class UIPlanner:
    """Training-free action planner over an accessibility/UI graph."""

    SAVE_WORDS=("save","kaydet")
    OPEN_WORDS=("open","aç")
    CLOSE_WORDS=("close","kapat")

    def plan(self,goal:str,graph:UIGraph)->list[UIAction]:
        g=goal.casefold()
        if any(x in g for x in self.SAVE_WORDS):
            candidates=graph.find("save")+graph.find("kaydet")
            if candidates:
                return [UIAction("CLICK",candidates[0].id)]
            return [UIAction("KEY",payload="CTRL+S")]

        type_match=re.search(
            r"(?:type|write|enter|yaz|gir)\s+[\"“']?(.+?)[\"”']?(?:\s+(?:into|to|içine|alanına))?$",
            goal,re.I,
        )
        payload=None
        if type_match:
            payload=type_match.group(1).strip(" .\"'“”")
        elif any(word in g for word in ("type","write","enter","yaz","gir")):
            quoted=re.search(r"[\"“'](.+?)[\"”']",goal)
            if quoted:
                payload=quoted.group(1).strip()
        if payload:
            boxes=graph.find(roles=("textbox","input","editor"))
            if boxes:
                return [UIAction("CLICK",boxes[0].id),UIAction("TYPE",payload=payload)]
            return [UIAction("TYPE",payload=payload)]

        for word in self.OPEN_WORDS:
            if word in g:
                target=g.split(word,1)[-1].strip(" :.'\"")
                hits=graph.find(target) if target else []
                if hits:
                    return [UIAction("DOUBLE_CLICK",hits[0].id)]

        if "scroll" in g or "kaydır" in g:
            direction="UP" if ("up" in g or "yukarı" in g) else "DOWN"
            return [UIAction("SCROLL",payload=direction)]

        return [UIAction("OBSERVE")]
