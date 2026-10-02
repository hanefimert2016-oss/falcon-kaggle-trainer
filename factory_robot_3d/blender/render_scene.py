from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from factory_robot_3d.blender.render_config import configure_cycles


def _script_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import bpy

    args = _args(_script_args() if argv is None else argv)
    frames_dir = args.frames_dir.resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    selected = configure_cycles(scene, bpy_module=bpy)
    scene.render.filepath = str(frames_dir / "frame_")
    scene.render.image_settings.file_format = "PNG"

    payload = {
        "backend": selected.backend,
        "device_names": list(selected.device_names),
        "engine": scene.render.engine,
        "cycles_device": scene.cycles.device,
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "filepath": scene.render.filepath,
    }
    print(
        "FACTORY_RENDER_RUNTIME_DEVICE " + json.dumps(payload, sort_keys=True),
        flush=True,
    )

    bpy.ops.render.render(animation=True)

    rendered = sorted(frames_dir.glob("frame_*.png"))
    print(
        f"FACTORY_RENDER_RUNTIME_COMPLETE frames={len(rendered)} dir={frames_dir}",
        flush=True,
    )
    if len(rendered) != 288:
        raise RuntimeError(f"expected 288 rendered frames, got {len(rendered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
