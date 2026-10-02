from __future__ import annotations

import argparse
from pathlib import Path
import sys

from factory_robot_3d.config import FactoryConfig
from factory_robot_3d.layout import build_factory_layout
from factory_robot_3d.blender.animation import apply_animation
from factory_robot_3d.blender.cameras import build_camera_plan
from factory_robot_3d.blender.factory_scene import build_factory_shell
from factory_robot_3d.blender.lighting import configure_factory_lighting
from factory_robot_3d.blender.materials import ensure_materials
from factory_robot_3d.blender.robot_rig import build_robot_manifest, create_robot_rig
from factory_robot_3d.blender.schema import load_animation_bundle


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _script_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)


def main(argv: list[str] | None = None) -> int:
    import bpy

    args = _args(_script_args() if argv is None else argv)
    bundle = load_animation_bundle(args.input)
    clear_scene(bpy)

    build_factory_shell(bpy.context.scene, bundle, bpy_module=bpy)
    materials = ensure_materials(bpy)

    layout = build_factory_layout(FactoryConfig.cinematic_default())
    ids = tuple(robot.robot_id for robot in layout.robots)
    bases = tuple(
        (
            float(robot.base_position[0]),
            float(robot.base_position[1]),
            float(robot.base_position[2] + 0.28),
            float(robot.base_yaw),
        )
        for robot in layout.robots
    )
    manifest = build_robot_manifest(ids, bases)

    collection = bpy.data.collections.get("FactoryRobots")
    if collection is None:
        collection = bpy.data.collections.new("FactoryRobots")
        bpy.context.scene.collection.children.link(collection)

    rigs = {
        robot_id: create_robot_rig(
            robot_id,
            manifest.rigs[robot_id].base_transform,
            materials=materials,
            bpy_module=bpy,
            collection=collection,
        )
        for robot_id in ids
    }

    configure_factory_lighting(bpy.context.scene, bpy_module=bpy)
    animation_manifest = apply_animation(
        bundle,
        rigs,
        bpy_module=bpy,
        materials=materials,
    )
    camera_plan = build_camera_plan(bpy.context.scene, bpy_module=bpy)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 288
    bpy.context.scene.render.fps = 24
    bpy.context.scene["factory_robot_count"] = len(rigs)
    bpy.context.scene["factory_cell_count"] = 5
    bpy.context.scene["factory_animation_frames"] = animation_manifest.frame_count
    bpy.context.scene["factory_camera_shots"] = len(camera_plan.shots)
    bpy.context.scene["factory_smoke"] = bool(args.smoke)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"FACTORY_SCENE_OK robots={len(rigs)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
