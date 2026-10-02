from __future__ import annotations

import json

from factory_robot_3d.blender.render_config import configure_cycles


def main() -> int:
    import bpy

    selected = configure_cycles(bpy.context.scene, bpy_module=bpy)
    payload = {
        "backend": selected.backend,
        "device_names": list(selected.device_names),
        "engine": bpy.context.scene.render.engine,
        "cycles_device": bpy.context.scene.cycles.device,
    }
    print("FACTORY_RENDER_REINIT " + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
