from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factory_robot_3d.blender.render_config import configure_cycles


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _script_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def main(argv: list[str] | None = None) -> int:
    import bpy

    args = _args(_script_args() if argv is None else argv)
    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 288
    scene.render.fps = 24
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_file_extension = True
    scene.render.filepath = str(frames_dir / "frame_####")
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True

    selected = configure_cycles(scene, bpy_module=bpy)
    payload = {
        "backend": selected.backend,
        "device_names": list(selected.device_names),
        "engine": scene.render.engine,
        "cycles_device": scene.cycles.device,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "output": scene.render.filepath,
    }
    print(
        "FACTORY_RENDER_RUNTIME_DEVICE " + json.dumps(payload, sort_keys=True),
        flush=True,
    )

    total_frames = scene.frame_end - scene.frame_start + 1
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        target = frames_dir / f"frame_{frame:04d}"
        scene.render.filepath = str(target)
        print(
            f"FACTORY_FRAME_BEGIN frame={frame}/{scene.frame_end}",
            flush=True,
        )
        bpy.ops.render.render(write_still=True)
        rendered = target.with_suffix(".png")
        if not rendered.is_file() or rendered.stat().st_size == 0:
            raise RuntimeError(f"rendered frame missing or empty: {rendered}")
        print(
            f"FACTORY_FRAME_OK frame={frame}/{scene.frame_end} "
            f"bytes={rendered.stat().st_size}",
            flush=True,
        )

    first = frames_dir / "frame_0001.png"
    last = frames_dir / "frame_0288.png"
    if not first.is_file() or not last.is_file():
        raise RuntimeError(
            f"animation render incomplete: first={first.exists()} last={last.exists()}"
        )

    print(
        "FACTORY_RENDER_ANIMATION_OK "
        f"first={first} last={last}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
