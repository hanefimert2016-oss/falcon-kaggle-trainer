from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Protocol

from .action_protocol import Action, ActionType


@dataclass(frozen=True)
class ScreenGeometry:
    width: int
    height: int

    def pixel(self, x: float, y: float) -> tuple[int, int]:
        px = round(min(1.0, max(0.0, x)) * max(0, self.width - 1))
        py = round(min(1.0, max(0.0, y)) * max(0, self.height - 1))
        return px, py


class InputBackend(Protocol):
    def position(self) -> tuple[int, int]: ...
    def move_to(self, x: int, y: int, duration: float = 0.0) -> None: ...
    def click(self, button: str = "left", clicks: int = 1, interval: float = 0.0) -> None: ...
    def drag_to(self, x: int, y: int, duration: float, button: str = "left") -> None: ...
    def scroll(self, x: int, y: int) -> None: ...
    def write(self, text: str, interval: float) -> None: ...
    def hotkey(self, *keys: str) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...


class DryRunBackend:
    """Deterministic backend used by tests and safe previews."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self._pos = (0, 0)

    def position(self): return self._pos
    def move_to(self, x, y, duration=0.0):
        self._pos = (int(x), int(y)); self.events.append(("move_to", int(x), int(y), float(duration)))
    def click(self, button="left", clicks=1, interval=0.0):
        self.events.append(("click", button, int(clicks), float(interval)))
    def drag_to(self, x, y, duration, button="left"):
        self._pos = (int(x), int(y)); self.events.append(("drag_to", int(x), int(y), float(duration), button))
    def scroll(self, x, y): self.events.append(("scroll", int(x), int(y)))
    def write(self, text, interval): self.events.append(("write", str(text), float(interval)))
    def hotkey(self, *keys): self.events.append(("hotkey", *keys))
    def key_down(self, key): self.events.append(("key_down", key))
    def key_up(self, key): self.events.append(("key_up", key))


class PyAutoGUIBackend:
    """Real OS input backend. Import is lazy so training/CI never needs a desktop."""

    def __init__(self, fail_safe: bool = True) -> None:
        import pyautogui
        pyautogui.FAILSAFE = fail_safe
        pyautogui.PAUSE = 0
        self.pg = pyautogui

    def position(self):
        p = self.pg.position()
        return int(p.x), int(p.y)
    def move_to(self, x, y, duration=0.0): self.pg.moveTo(x, y, duration=duration)
    def click(self, button="left", clicks=1, interval=0.0): self.pg.click(button=button, clicks=clicks, interval=interval)
    def drag_to(self, x, y, duration, button="left"): self.pg.dragTo(x, y, duration=duration, button=button)
    def scroll(self, x, y):
        if x:
            self.pg.hscroll(x)
        if y:
            self.pg.scroll(y)
    def write(self, text, interval): self.pg.write(text, interval=interval)
    def hotkey(self, *keys): self.pg.hotkey(*keys)
    def key_down(self, key): self.pg.keyDown(key)
    def key_up(self, key): self.pg.keyUp(key)


class HumanInputExecutor:
    """Turn normalized model actions into real mouse/keyboard events.

    Mouse motion follows a short curved path instead of teleporting. Typing uses
    small randomized inter-key delays. Set sleep_enabled=False for unit tests.
    """

    def __init__(
        self,
        backend: InputBackend,
        geometry: ScreenGeometry,
        *,
        seed: int = 606,
        sleep_enabled: bool = True,
    ) -> None:
        if geometry.width <= 0 or geometry.height <= 0:
            raise ValueError("invalid screen geometry")
        self.backend = backend
        self.geometry = geometry
        self.rng = random.Random(seed)
        self.sleep_enabled = sleep_enabled

    def _sleep(self, seconds: float) -> None:
        if self.sleep_enabled and seconds > 0:
            time.sleep(seconds)

    def _move_human(self, x: int, y: int, duration: float | None = None) -> None:
        sx, sy = self.backend.position()
        dist = math.hypot(x - sx, y - sy)
        total = duration if duration is not None else min(0.85, max(0.08, 0.08 + dist / 1800.0))
        steps = max(2, min(18, int(dist / 80) + 3))
        # Quadratic Bezier with a small perpendicular bend.
        dx, dy = x - sx, y - sy
        norm = max(1.0, math.hypot(dx, dy))
        bend = min(45.0, dist * 0.08) * self.rng.uniform(-1.0, 1.0)
        cx = (sx + x) / 2.0 - dy / norm * bend
        cy = (sy + y) / 2.0 + dx / norm * bend
        for i in range(1, steps + 1):
            t = i / steps
            omt = 1.0 - t
            px = round(omt * omt * sx + 2 * omt * t * cx + t * t * x)
            py = round(omt * omt * sy + 2 * omt * t * cy + t * t * y)
            self.backend.move_to(px, py, duration=total / steps)

    def execute(self, action: Action) -> dict:
        action = action.validate()
        t = action.type
        if t is ActionType.DONE:
            return {"done": True, "action": action.to_dict()}

        if t in {ActionType.MOVE, ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK}:
            x, y = self.geometry.pixel(float(action.x), float(action.y))
            self._move_human(x, y, action.duration)
            if t is ActionType.CLICK:
                self.backend.click(button=action.button, clicks=1)
            elif t is ActionType.DOUBLE_CLICK:
                self.backend.click(button=action.button, clicks=2, interval=self.rng.uniform(0.05, 0.12))
            elif t is ActionType.RIGHT_CLICK:
                self.backend.click(button="right", clicks=1)

        elif t is ActionType.DRAG:
            x1, y1 = self.geometry.pixel(float(action.x), float(action.y))
            x2, y2 = self.geometry.pixel(float(action.x2), float(action.y2))
            self._move_human(x1, y1, action.duration)
            self.backend.drag_to(x2, y2, duration=action.duration or self.rng.uniform(0.25, 0.65), button=action.button)

        elif t is ActionType.SCROLL:
            self.backend.scroll(action.scroll_x, action.scroll_y)

        elif t is ActionType.TYPE:
            # pyautogui.write is reliable for ASCII. A platform adapter can
            # replace the backend for richer Unicode input on Wayland/X11.
            interval = self.rng.uniform(0.018, 0.055)
            self.backend.write(action.text, interval=interval)

        elif t is ActionType.KEY:
            self.backend.hotkey(*action.keys)

        elif t is ActionType.KEY_DOWN:
            for key in action.keys:
                self.backend.key_down(key)

        elif t is ActionType.KEY_UP:
            for key in reversed(action.keys):
                self.backend.key_up(key)

        elif t is ActionType.WAIT:
            self._sleep(float(action.seconds))

        else:
            raise ValueError(f"unsupported action type: {t}")

        self._sleep(self.rng.uniform(0.025, 0.11))
        return {"done": False, "action": action.to_dict()}
