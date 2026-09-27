from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Protocol

import numpy as np
from PIL import Image
import torch
from tokenizers import Tokenizer

from flm.computer_use.action_protocol import Action, ActionType
from flm.computer_use.executor import (
    DryRunBackend,
    HumanInputExecutor,
    PyAutoGUIBackend,
    ScreenGeometry,
)
from flm.models.computer_use_v3 import ComputerUseV3, ComputerUseV3Config


OPS_V3 = [
    "MOVE",
    "CLICK",
    "DOUBLE_CLICK",
    "RIGHT_CLICK",
    "DRAG",
    "SCROLL",
    "TYPE",
    "KEY",
    "KEY_DOWN",
    "KEY_UP",
    "WAIT",
    "DONE",
]


class CaptureBackend(Protocol):
    def capture(self) -> Image.Image: ...
    def geometry(self) -> ScreenGeometry: ...


class MSSCapture:
    """Fast cross-platform screen capture.

    On Linux, the session must allow screenshot capture (X11 or a compositor /
    portal setup that mss can access).
    """

    def __init__(self, monitor: int = 1):
        import mss
        self._mss = mss.mss()
        if monitor < 0 or monitor >= len(self._mss.monitors):
            raise ValueError(f"invalid monitor {monitor}; available 0..{len(self._mss.monitors)-1}")
        self._monitor = monitor

    def geometry(self) -> ScreenGeometry:
        m = self._mss.monitors[self._monitor]
        return ScreenGeometry(int(m["width"]), int(m["height"]))

    def capture(self) -> Image.Image:
        shot = self._mss.grab(self._mss.monitors[self._monitor])
        return Image.frombytes("RGB", shot.size, shot.rgb)


@dataclass
class PolicyPrediction:
    action: Action
    op_id: int
    domain_id: int
    payload: str


class ComputerUsePolicy:
    def __init__(self, checkpoint: str | Path, tokenizer: str | Path, device: str | None = None):
        ck = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        cfg = ComputerUseV3Config(**ck["config"])
        model = ComputerUseV3(cfg)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.cfg = cfg
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.domains = ck.get("domains", {})
        self.id_to_domain = {int(v): k for k, v in self.domains.items()}
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            raise RuntimeError("tokenizer does not contain <pad>")

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        # Direct resize preserves normalized coordinates exactly. The model sees
        # the same mapping during training.
        im = image.convert("RGB").resize((self.cfg.image_size, self.cfg.image_size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)

    def _task_tensor(self, task: str) -> torch.Tensor:
        ids = self.tokenizer.encode(str(task), add_special_tokens=False).ids[: self.cfg.task_len]
        ids += [self.pad_id] * (self.cfg.task_len - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    @staticmethod
    def _keys(payload: str) -> tuple[str, ...]:
        return tuple(x.strip() for x in re.split(r"[+,]", payload) if x.strip())

    @staticmethod
    def _scroll(payload: str) -> tuple[int, int]:
        nums = re.findall(r"-?\d+", payload)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        if len(nums) == 1:
            return 0, int(nums[0])
        return 0, 0

    def _to_action(self, op_id: int, coord, coord2, payload: str) -> Action:
        if not (0 <= op_id < len(OPS_V3)):
            raise ValueError(f"invalid predicted op id {op_id}")
        name = OPS_V3[op_id]
        x, y = float(coord[0]), float(coord[1])
        x2, y2 = float(coord2[0]), float(coord2[1])

        if name == "MOVE":
            return Action(ActionType.MOVE, x=x, y=y).validate()
        if name == "CLICK":
            return Action(ActionType.CLICK, x=x, y=y).validate()
        if name == "DOUBLE_CLICK":
            return Action(ActionType.DOUBLE_CLICK, x=x, y=y).validate()
        if name == "RIGHT_CLICK":
            return Action(ActionType.RIGHT_CLICK, x=x, y=y).validate()
        if name == "DRAG":
            return Action(ActionType.DRAG, x=x, y=y, x2=x2, y2=y2).validate()
        if name == "SCROLL":
            sx, sy = self._scroll(payload)
            return Action(ActionType.SCROLL, scroll_x=sx, scroll_y=sy).validate()
        if name == "TYPE":
            text = payload.strip()
            if not text:
                # A malformed empty payload is safer as WAIT than an arbitrary key.
                return Action(ActionType.WAIT, seconds=0.15).validate()
            return Action(ActionType.TYPE, text=text).validate()
        if name == "KEY":
            keys = self._keys(payload)
            if not keys:
                return Action(ActionType.WAIT, seconds=0.15).validate()
            return Action(ActionType.KEY, keys=keys).validate()
        if name == "KEY_DOWN":
            keys = self._keys(payload)
            if not keys:
                return Action(ActionType.WAIT, seconds=0.15).validate()
            return Action(ActionType.KEY_DOWN, keys=keys).validate()
        if name == "KEY_UP":
            keys = self._keys(payload)
            if not keys:
                return Action(ActionType.WAIT, seconds=0.15).validate()
            return Action(ActionType.KEY_UP, keys=keys).validate()
        if name == "WAIT":
            nums = re.findall(r"\d+(?:\.\d+)?", payload)
            seconds = min(5.0, max(0.05, float(nums[0]) if nums else 0.25))
            return Action(ActionType.WAIT, seconds=seconds).validate()
        return Action(ActionType.DONE).validate()

    @torch.inference_mode()
    def predict(self, image: Image.Image, task: str) -> PolicyPrediction:
        image_t = self._image_tensor(image).to(self.device)
        task_t = self._task_tensor(task).to(self.device)
        raw = self.model.predict(image_t, task_t)
        op_id = int(raw["op"][0].item())
        domain_id = int(raw["domain"][0].item())
        coord = raw["coord"][0].float().cpu().tolist()
        coord2 = raw["coord2"][0].float().cpu().tolist()
        payload = str(raw["payload"][0])
        return PolicyPrediction(
            action=self._to_action(op_id, coord, coord2, payload),
            op_id=op_id,
            domain_id=domain_id,
            payload=payload,
        )


class ComputerUseAgent:
    """Observe -> predict one action -> execute -> observe again."""

    def __init__(
        self,
        policy: ComputerUsePolicy,
        capture: CaptureBackend,
        executor: HumanInputExecutor,
        *,
        max_steps: int = 80,
        settle_seconds: float = 0.18,
    ):
        self.policy = policy
        self.capture_backend = capture
        self.executor = executor
        self.max_steps = max_steps
        self.settle_seconds = settle_seconds

    def run(self, task: str) -> list[dict]:
        history = []
        previous_frame = None
        previous_action = None
        repeated_stall = 0

        for step in range(self.max_steps):
            screenshot = self.capture_backend.capture()
            # A tiny grayscale preview is enough to detect an unchanged desktop
            # while ignoring most compression/noise differences.
            frame = np.asarray(
                screenshot.convert("L").resize((32, 32)),
                dtype=np.float32,
            ) / 255.0

            pred = self.policy.predict(screenshot, task)
            action_json = pred.action.to_json()
            screen_delta = (
                None
                if previous_frame is None
                else float(np.mean(np.abs(frame - previous_frame)))
            )
            if (
                previous_action == action_json
                and screen_delta is not None
                and screen_delta < 0.008
            ):
                repeated_stall += 1
            else:
                repeated_stall = 0

            event = {
                "step": step,
                "op": OPS_V3[pred.op_id],
                "domain": self.policy.id_to_domain.get(pred.domain_id, f"id:{pred.domain_id}"),
                "payload": pred.payload,
                "action": pred.action.to_dict(),
                "screen_delta": screen_delta,
                "repeated_stall": repeated_stall,
            }
            history.append(event)

            if repeated_stall >= 3:
                # Do not blindly hammer the same pixel/key forever when the UI
                # did not react. Give the application time to settle, then
                # re-observe. Abort after repeated failures so a caller can
                # re-plan or ask for help.
                wait = Action(ActionType.WAIT, seconds=0.5).validate()
                event["guard"] = "stalled_repeat"
                event["executed_action"] = wait.to_dict()
                result = self.executor.execute(wait)
                if repeated_stall >= 5:
                    event["guard"] = "stalled_abort"
                    return history
            else:
                result = self.executor.execute(pred.action)

            previous_frame = frame
            previous_action = action_json

            if result["done"]:
                return history
            if self.executor.sleep_enabled and self.settle_seconds:
                time.sleep(self.settle_seconds)
        return history


def build_desktop_agent(
    checkpoint: str | Path,
    tokenizer: str | Path,
    *,
    execute: bool = False,
    monitor: int = 1,
    max_steps: int = 80,
    device: str | None = None,
):
    capture = MSSCapture(monitor=monitor)
    geometry = capture.geometry()
    backend = PyAutoGUIBackend(fail_safe=True) if execute else DryRunBackend()
    executor = HumanInputExecutor(
        backend,
        geometry,
        sleep_enabled=execute,
    )
    policy = ComputerUsePolicy(checkpoint, tokenizer, device=device)
    return ComputerUseAgent(policy, capture, executor, max_steps=max_steps)
