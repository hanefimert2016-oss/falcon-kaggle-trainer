from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from typing import Any, Callable

from .protocol import ToolCall, extract_tool_calls, format_tool_result


@dataclass
class RegisteredTool:
    name: str
    fn: Callable[..., Any]
    description: str
    schema: dict[str, Any]


class ToolRegistry:
    """Explicit allow-list. FLM-Coder can only invoke tools registered here."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        description: str = "",
        schema: dict[str, Any] | None = None,
    ) -> None:
        ToolCall(name=name, arguments={}).validate()
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        if schema is None:
            sig = inspect.signature(fn)
            props = {}
            required = []
            for p in sig.parameters.values():
                props[p.name] = {"type": "string"}
                if p.default is inspect.Parameter.empty:
                    required.append(p.name)
            schema = {"type": "object", "properties": props, "required": required}
        self._tools[name] = RegisteredTool(name, fn, description, schema)

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.schema}
            for t in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> Any:
        call.validate()
        if call.name not in self._tools:
            raise KeyError(f"tool not allowed: {call.name}")
        return self._tools[call.name].fn(**call.arguments)


def tool_system_prompt(registry: ToolRegistry) -> str:
    return (
        "Available tools (JSON schema):\n"
        + json.dumps(registry.specs(), ensure_ascii=False, separators=(",", ":"))
        + "\nWhen a tool is needed, emit exactly "
          "<|tool_call|>{\"name\":\"tool_name\",\"arguments\":{...}}<|tool_end|>. "
          "Otherwise answer normally."
    )


def run_tool_loop(
    generate: Callable[[list[dict[str, str]]], str],
    messages: list[dict[str, str]],
    registry: ToolRegistry,
    *,
    max_rounds: int = 8,
) -> tuple[str, list[dict[str, str]]]:
    history = list(messages)
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": tool_system_prompt(registry)})
    else:
        history[0] = {
            **history[0],
            "content": history[0].get("content", "") + "\n\n" + tool_system_prompt(registry),
        }

    for _ in range(max_rounds):
        output = generate(history)
        history.append({"role": "assistant", "content": output})
        calls = extract_tool_calls(output)
        if not calls:
            return output, history
        for call in calls:
            try:
                result = registry.execute(call)
                payload = format_tool_result(call.name, result, call_id=call.call_id)
            except Exception as exc:
                payload = format_tool_result(
                    call.name,
                    {"error": type(exc).__name__, "message": str(exc)},
                    call_id=call.call_id,
                )
            history.append({"role": "tool", "content": payload})
    raise RuntimeError("tool loop exceeded max_rounds")
