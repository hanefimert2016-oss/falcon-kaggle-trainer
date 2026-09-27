from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

TOOL_CALL_START = "<|tool_call|>"
TOOL_RESULT_START = "<|tool_result|>"
TOOL_END = "<|tool_end|>"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None

    def validate(self) -> "ToolCall":
        if not self.name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}", self.name):
            raise ValueError(f"invalid tool name: {self.name!r}")
        if not isinstance(self.arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        return self


def format_tool_call(call: ToolCall) -> str:
    call.validate()
    payload: dict[str, Any] = {"name": call.name, "arguments": call.arguments}
    if call.call_id:
        payload["id"] = call.call_id
    return TOOL_CALL_START + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + TOOL_END


def format_tool_result(name: str, result: Any, *, call_id: str | None = None) -> str:
    payload: dict[str, Any] = {"name": name, "result": result}
    if call_id:
        payload["id"] = call_id
    return TOOL_RESULT_START + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str) + TOOL_END


def extract_tool_calls(text: str) -> list[ToolCall]:
    pattern = re.compile(re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_END), re.S)
    calls: list[ToolCall] = []
    for raw in pattern.findall(text or ""):
        obj = json.loads(raw.strip())
        objs = obj if isinstance(obj, list) else [obj]
        for item in objs:
            if not isinstance(item, dict):
                raise ValueError("tool call payload must be an object or list of objects")
            args = item.get("arguments", item.get("args", {}))
            if isinstance(args, str):
                args = json.loads(args)
            call = ToolCall(
                name=str(item.get("name") or ""),
                arguments=args,
                call_id=str(item["id"]) if item.get("id") is not None else None,
            ).validate()
            calls.append(call)
    return calls
