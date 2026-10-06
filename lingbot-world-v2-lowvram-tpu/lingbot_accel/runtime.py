from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Accelerator:
    kind: str
    device: torch.device
    xla: object | None = None

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    @property
    def is_xla(self) -> bool:
        return self.kind == "xla"

    def mark_step(self) -> None:
        if self.is_xla and self.xla is not None:
            self.xla.mark_step()

    def synchronize(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        elif self.is_xla and self.xla is not None:
            self.xla.mark_step()
            self.xla.wait_device_ops()

    def empty_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()

    @contextlib.contextmanager
    def autocast(self, dtype: torch.dtype):
        if self.is_cuda:
            with torch.amp.autocast("cuda", dtype=dtype):
                yield
        else:
            yield


def detect_accelerator(prefer: str = "auto", cuda_index: int = 0) -> Accelerator:
    prefer = (prefer or "auto").lower()
    if prefer not in {"auto", "cuda", "xla", "cpu"}:
        raise ValueError(f"Unsupported accelerator: {prefer}")

    if prefer in {"auto", "xla"}:
        try:
            import torch_xla.core.xla_model as xm
            dev = xm.xla_device()
            if prefer == "xla" or str(dev).startswith("xla"):
                return Accelerator("xla", dev, xm)
        except Exception:
            if prefer == "xla":
                raise

    if prefer in {"auto", "cuda"} and torch.cuda.is_available():
        return Accelerator("cuda", torch.device(f"cuda:{cuda_index}"))

    if prefer == "cuda":
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    return Accelerator("cpu", torch.device("cpu"))


def seed_everything(seed: int, acc: Accelerator) -> Optional[torch.Generator]:
    torch.manual_seed(seed)
    if acc.is_cuda:
        g = torch.Generator(device=acc.device)
        g.manual_seed(seed)
        return g
    return None


def device_memory_gb(acc: Accelerator) -> float | None:
    if acc.is_cuda:
        props = torch.cuda.get_device_properties(acc.device)
        return props.total_memory / (1024 ** 3)
    return None


def now_sync(acc: Accelerator) -> float:
    acc.synchronize()
    return time.perf_counter()
