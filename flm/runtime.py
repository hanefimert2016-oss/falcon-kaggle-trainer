from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import torch

@dataclass
class Runtime:
    kind: str
    device: torch.device
    world_size: int
    gpu_names: list[str]
    xm: object | None = None

    def optimizer_step(self, optimizer, scaler=None) -> None:
        if self.kind == "tpu":
            assert self.xm is not None
            self.xm.optimizer_step(optimizer, barrier=True)
        elif scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    def autocast(self):
        if self.kind == "gpu":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

def select_runtime(requested: str | None = None) -> Runtime:
    requested = (requested or os.environ.get("FLM_ACCELERATOR", "auto")).lower()
    if requested in {"tpu", "auto"}:
        try:
            import torch_xla.core.xla_model as xm
            if requested == "tpu" or os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
                return Runtime("tpu", xm.xla_device(), 1, ["TPU/XLA"], xm=xm)
        except Exception:
            if requested == "tpu":
                raise
    if requested in {"gpu", "cuda", "auto"} and torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return Runtime("gpu", torch.device("cuda"), torch.cuda.device_count(), names)
    if requested in {"gpu", "cuda", "tpu"}:
        raise RuntimeError(f"requested accelerator {requested!r} is unavailable")
    return Runtime("cpu", torch.device("cpu"), 1, [])
