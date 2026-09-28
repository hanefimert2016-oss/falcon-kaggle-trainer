from __future__ import annotations

from typing import Any


def summarize_action(
    operation: str,
    *,
    coord=None,
    coord2=None,
    payload: str = "",
) -> str:
    """Compact, deterministic action history used by training and runtime."""
    op = str(operation or "OTHER").upper()
    if coord is not None and op in {"MOVE", "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"}:
        return f"{op}@{float(coord[0]):.3f},{float(coord[1]):.3f}"
    if coord is not None and coord2 is not None and op == "DRAG":
        return (
            f"DRAG@{float(coord[0]):.3f},{float(coord[1]):.3f}"
            f"->{float(coord2[0]):.3f},{float(coord2[1]):.3f}"
        )
    text = str(payload or "").replace("\n", " ").strip()
    if len(text) > 64:
        text = text[:61] + "..."
    return f"{op}:{text}" if text else op


def summarize_action_dict(action: dict[str, Any]) -> str:
    op = str(action.get("type") or "OTHER").upper()
    payload = ""
    if op == "TYPE":
        payload = str(action.get("text") or "")
    elif op in {"KEY", "KEY_DOWN", "KEY_UP"}:
        payload = "+".join(str(x) for x in (action.get("keys") or []))
    elif op == "SCROLL":
        payload = f"{int(action.get('scroll_x') or 0)},{int(action.get('scroll_y') or 0)}"
    elif op == "WAIT":
        payload = str(action.get("seconds") or "")
    coord = None
    coord2 = None
    if action.get("x") is not None and action.get("y") is not None:
        coord = [action["x"], action["y"]]
    if action.get("x2") is not None and action.get("y2") is not None:
        coord2 = [action["x2"], action["y2"]]
    return summarize_action(op, coord=coord, coord2=coord2, payload=payload)


def build_context_ids(tok, task: str, history, length: int, pad_id: int):
    """Encode task + recent action history while guaranteeing both fit.

    The newest history tokens are retained, and at least half of the context is
    reserved for the original task so long instructions do not erase state.
    """
    if length <= 0:
        raise ValueError("context length must be positive")
    task_ids = tok.encode("Task: " + str(task), add_special_tokens=False).ids
    hist = [str(x) for x in (history or []) if str(x).strip()][-8:]
    if hist:
        marker_ids = tok.encode(
            "\nRecent actions:",
            add_special_tokens=False,
        ).ids
        action_ids = tok.encode(
            " " + " | ".join(hist),
            add_special_tokens=False,
        ).ids
        hist_budget = min(
            len(marker_ids) + len(action_ids),
            max(len(marker_ids) + 16, length // 2),
        )
        # Preserve the semantic marker and keep the newest action tokens.
        action_budget = max(0, hist_budget - len(marker_ids))
        if action_budget:
            action_ids = action_ids[-action_budget:]
        else:
            action_ids = []
        hist_ids = marker_ids + action_ids
        hist_ids = hist_ids[:length - 1] if length > 1 else []
        task_budget = max(1, length - len(hist_ids))
        ids = task_ids[:task_budget] + hist_ids
    else:
        ids = task_ids[:length]
    ids = ids[:length]
    ids += [pad_id] * (length - len(ids))
    return ids
