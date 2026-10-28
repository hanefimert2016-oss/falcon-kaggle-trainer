from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_REQUIRED_MATERIALS = {
    "PaintedMetal",
    "BrushedMetal",
    "Rubber",
    "SafetyYellow",
    "Glass",
    "EmissiveDisplay",
    "WarningRed",
    "EpoxyFloor",
}


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...]


def validate_scene_manifest(
    *,
    robot_count: int,
    cell_count: int,
    frame_count: int,
    camera_shot_count: int,
    material_names: Iterable[str],
    lighting_ok: bool,
) -> ValidationReport:
    errors: list[str] = []
    if robot_count != 20:
        errors.append(f"robot_count must be 20, got {robot_count}")
    if cell_count != 5:
        errors.append(f"cell_count must be 5, got {cell_count}")
    if frame_count != 288:
        errors.append(f"frame_count must be 288, got {frame_count}")
    if camera_shot_count != 6:
        errors.append(f"camera_shot_count must be 6, got {camera_shot_count}")
    missing = sorted(_REQUIRED_MATERIALS - set(material_names))
    if missing:
        errors.append("missing materials: " + ", ".join(missing))
    if not lighting_ok:
        errors.append("factory cinematic lighting is incomplete")
    return ValidationReport(ok=not errors, errors=tuple(errors))


def validate_scene(scene: Any, bundle: Any, *, bpy_module: Any | None = None) -> ValidationReport:
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore
    bpy = bpy_module

    material_names = set(bpy.data.materials.keys())
    lighting_collection = bpy.data.collections.get("FactoryLighting")
    lighting_ok = bool(
        lighting_collection is not None
        and len(lighting_collection.objects) >= 21
        and scene.world is not None
    )
    robot_collection = bpy.data.collections.get("FactoryRobots")
    robot_roots = []
    if robot_collection is not None:
        robot_roots = [
            obj for obj in robot_collection.objects
            if obj.name.startswith("Robot_R") and obj.name.endswith("_Root")
        ]

    report = validate_scene_manifest(
        robot_count=len(robot_roots),
        cell_count=int(scene.get("factory_cell_count", 0)),
        frame_count=int(scene.get("factory_animation_frames", 0)),
        camera_shot_count=int(scene.get("factory_camera_shots", 0)),
        material_names=material_names,
        lighting_ok=lighting_ok,
    )

    extra_errors = list(report.errors)
    if len(bundle.frames) != 288:
        extra_errors.append("animation bundle must contain 288 frames")
    if scene.render.engine != "CYCLES":
        extra_errors.append("render engine must be CYCLES")
    backend = str(scene.get("factory_render_backend", ""))
    if backend not in {"OPTIX", "CUDA"}:
        extra_errors.append("production render backend must be OPTIX or CUDA")

    return ValidationReport(ok=not extra_errors, errors=tuple(extra_errors))
