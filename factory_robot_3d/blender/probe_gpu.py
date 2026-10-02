from __future__ import annotations

import json
import sys
from typing import Any, Iterable

from factory_robot_3d.blender.render_config import DeviceInfo, select_cycles_backend


def probe_payload(
    devices: Iterable[DeviceInfo],
    *,
    blender_version: str,
) -> dict[str, object]:
    selected = select_cycles_backend(devices)
    if not selected.device_names or any("TESLA T4" not in name.upper() for name in selected.device_names):
        raise RuntimeError("Blender production probe requires NVIDIA Tesla T4")
    return {
        "backend": selected.backend,
        "device_names": list(selected.device_names),
        "blender_version": str(blender_version),
    }


def _cycles_devices(bpy: Any) -> tuple[DeviceInfo, ...]:
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        raise RuntimeError("Cycles addon is unavailable")

    prefs = addon.preferences
    discovered: list[DeviceInfo] = []
    seen: set[tuple[str, str]] = set()

    # Ask Blender for both compute families. Some builds only expose devices
    # after compute_device_type has been selected at least once.
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception:
            continue
        for device in getattr(prefs, "devices", ()):
            name = str(getattr(device, "name", "")).strip()
            dtype = str(getattr(device, "type", "")).upper().strip()
            key = (name, dtype)
            if not name or key in seen:
                continue
            seen.add(key)
            discovered.append(
                DeviceInfo(
                    name=name,
                    backend=dtype,
                    enabled=bool(getattr(device, "use", True)),
                )
            )

    # Final fallback for Blender builds where compute_device_type is read-only.
    try:
        prefs.get_devices()
    except Exception:
        pass
    for device in getattr(prefs, "devices", ()):
        name = str(getattr(device, "name", "")).strip()
        dtype = str(getattr(device, "type", "")).upper().strip()
        key = (name, dtype)
        if name and key not in seen:
            seen.add(key)
            discovered.append(
                DeviceInfo(
                    name=name,
                    backend=dtype,
                    enabled=bool(getattr(device, "use", True)),
                )
            )

    return tuple(discovered)


def runtime_probe(bpy_module: Any | None = None) -> dict[str, object]:
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore
    bpy = bpy_module
    devices = _cycles_devices(bpy)
    version = ".".join(str(x) for x in bpy.app.version)
    return probe_payload(devices, blender_version=version)


def main() -> int:
    payload = runtime_probe()
    line = "FACTORY_GPU_PROBE " + json.dumps(payload, sort_keys=True)
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
