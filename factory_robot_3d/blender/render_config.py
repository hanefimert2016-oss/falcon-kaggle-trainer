from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RenderContract:
    engine: str
    width: int
    height: int
    fps: int
    frame_start: int
    frame_end: int
    samples: int
    adaptive_sampling: bool
    denoising: bool
    motion_blur: bool
    view_transform: str


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    backend: str
    enabled: bool = True


@dataclass(frozen=True)
class RenderDeviceInfo:
    backend: str
    device_names: tuple[str, ...]


def production_render_contract() -> RenderContract:
    return RenderContract(
        engine="CYCLES",
        width=1920,
        height=1080,
        fps=24,
        frame_start=1,
        frame_end=288,
        samples=64,
        adaptive_sampling=True,
        denoising=True,
        motion_blur=True,
        view_transform="AgX",
    )


def select_cycles_backend(devices: Iterable[DeviceInfo]) -> RenderDeviceInfo:
    normalized = tuple(devices)
    for backend in ("OPTIX", "CUDA"):
        names = tuple(
            device.name
            for device in normalized
            if device.enabled
            and device.backend.upper() == backend
            and "TESLA T4" in device.name.upper()
        )
        if names:
            return RenderDeviceInfo(backend=backend, device_names=names)
    raise RuntimeError("production render requires an NVIDIA Tesla T4 GPU using OPTIX or CUDA")


def _discover_cycles_devices(bpy: Any) -> tuple[Any, tuple[DeviceInfo, ...]]:
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        raise RuntimeError("Cycles addon is unavailable")
    prefs = addon.preferences
    try:
        prefs.get_devices()
    except Exception:
        pass

    infos: list[DeviceInfo] = []
    for device in getattr(prefs, "devices", ()):
        name = str(getattr(device, "name", ""))
        backend = str(getattr(device, "type", "")).upper()
        infos.append(DeviceInfo(name=name, backend=backend, enabled=True))
    return prefs, tuple(infos)


def configure_cycles(
    scene: Any,
    preferred: tuple[str, ...] = ("OPTIX", "CUDA"),
    *,
    bpy_module: Any | None = None,
) -> RenderDeviceInfo:
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore
    bpy = bpy_module

    contract = production_render_contract()
    scene.render.engine = contract.engine
    scene.render.resolution_x = contract.width
    scene.render.resolution_y = contract.height
    scene.render.resolution_percentage = 100
    scene.render.fps = contract.fps
    scene.frame_start = contract.frame_start
    scene.frame_end = contract.frame_end
    scene.cycles.samples = contract.samples
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = contract.adaptive_sampling
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = contract.denoising
    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = contract.motion_blur
    scene.view_settings.view_transform = contract.view_transform

    prefs, discovered = _discover_cycles_devices(bpy)
    available_backends = {device.backend for device in discovered}
    selected: RenderDeviceInfo | None = None

    # Request preferred backends explicitly because Cycles can expose only the
    # currently selected compute family in prefs.devices.
    for backend in preferred:
        backend = backend.upper()
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception:
            continue
        live = tuple(
            DeviceInfo(
                name=str(getattr(device, "name", "")),
                backend=str(getattr(device, "type", "")).upper(),
                enabled=True,
            )
            for device in getattr(prefs, "devices", ())
        )
        try:
            candidate = select_cycles_backend(live)
        except RuntimeError:
            continue
        if candidate.backend != backend:
            continue
        selected = candidate
        for device in getattr(prefs, "devices", ()):
            device_type = str(getattr(device, "type", "")).upper()
            device_name = str(getattr(device, "name", ""))
            device.use = bool(
                device_type == backend and "TESLA T4" in device_name.upper()
            )
        break

    if selected is None:
        # Last chance: already-discovered list, useful for mocked tests and
        # some Blender builds where assigning compute_device_type is read-only.
        selected = select_cycles_backend(discovered)

    scene.cycles.device = "GPU"
    scene["factory_render_backend"] = selected.backend
    scene["factory_render_devices"] = ",".join(selected.device_names)
    return selected
