from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import re
from typing import Any


class ActionType(str, Enum):
    MOVE = "move"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    KEY = "key"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    WAIT = "wait"
    DONE = "done"


@dataclass(frozen=True)
class Action:
    type: ActionType
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    button: str = "left"
    text: str = ""
    keys: tuple[str, ...] = ()
    scroll_x: int = 0
    scroll_y: int = 0
    duration: float | None = None
    seconds: float | None = None

    def validate(self) -> "Action":
        coords = (("x", self.x), ("y", self.y), ("x2", self.x2), ("y2", self.y2))
        for name, value in coords:
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be normalized to [0,1], got {value}")
        if self.type in {
            ActionType.MOVE, ActionType.CLICK, ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
        } and (self.x is None or self.y is None):
            raise ValueError(f"{self.type.value} requires x/y")
        if self.type is ActionType.DRAG and None in (self.x, self.y, self.x2, self.y2):
            raise ValueError("drag requires x/y and x2/y2")
        if self.type is ActionType.TYPE and not self.text:
            raise ValueError("type requires non-empty text")
        if self.type in {ActionType.KEY, ActionType.KEY_DOWN, ActionType.KEY_UP} and not self.keys:
            raise ValueError(f"{self.type.value} requires keys")
        if self.type is ActionType.WAIT:
            if self.seconds is None or not (0.0 <= self.seconds <= 30.0):
                raise ValueError("wait seconds must be in [0,30]")
        if self.duration is not None and not (0.0 <= self.duration <= 10.0):
            raise ValueError("duration must be in [0,10]")
        return self

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["type"] = self.type.value
        out["keys"] = list(self.keys)
        return {k: v for k, v in out.items() if v not in (None, "", (), [], 0) or k in ("type", "button")}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Action":
        kind = ActionType(str(raw.get("type") or raw.get("action") or "").strip().lower())
        keys = raw.get("keys", ())
        if isinstance(keys, str):
            keys = tuple(k.strip() for k in re.split(r"[+,]", keys) if k.strip())
        else:
            keys = tuple(str(k) for k in (keys or ()))
        action = cls(
            type=kind,
            x=_float_or_none(raw.get("x")),
            y=_float_or_none(raw.get("y")),
            x2=_float_or_none(raw.get("x2")),
            y2=_float_or_none(raw.get("y2")),
            button=str(raw.get("button") or "left").lower(),
            text=str(raw.get("text") or ""),
            keys=keys,
            scroll_x=int(raw.get("scroll_x") or raw.get("dx") or 0),
            scroll_y=int(raw.get("scroll_y") or raw.get("dy") or raw.get("amount") or 0),
            duration=_float_or_none(raw.get("duration")),
            seconds=_float_or_none(raw.get("seconds")),
        )
        return action.validate()


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_action(raw: str | dict[str, Any] | Action) -> Action:
    if isinstance(raw, Action):
        return raw.validate()
    if isinstance(raw, dict):
        return Action.from_mapping(raw)
    text = str(raw).strip()
    text = re.sub(r"^\s*<\|action\|>\s*", "", text)
    text = re.sub(r"\s*<\|action_end\|>\s*$", "", text)
    if text.startswith("```"):
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("computer-use action must be one JSON object")
    return Action.from_mapping(obj)
