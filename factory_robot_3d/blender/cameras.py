from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .animation import _animation_fcurves


@dataclass(frozen=True)
class CameraShot:
    name: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class CameraPlan:
    shots: tuple[CameraShot, ...]
    total_frames: int

    def validate(self) -> None:
        if not self.shots:
            raise ValueError("camera plan must contain shots")
        expected = 1
        for shot in self.shots:
            if shot.start_frame != expected or shot.end_frame < shot.start_frame:
                raise ValueError("camera shot ranges must be contiguous with no gaps or overlaps")
            expected = shot.end_frame + 1
        if expected - 1 != self.total_frames:
            raise ValueError("camera shot ranges must cover the full frame range")


def cinematic_camera_plan() -> CameraPlan:
    plan = CameraPlan(
        shots=(
            CameraShot("establishing", 1, 48),
            CameraShot("conveyor_tracking", 49, 96),
            CameraShot("synchronized_close", 97, 144),
            CameraShot("overhead_coordination", 145, 192),
            CameraShot("assembly_packaging_detail", 193, 240),
            CameraShot("final_hero", 241, 288),
        ),
        total_frames=288,
    )
    plan.validate()
    return plan


_SHOT_POSES = {
    "establishing": ((-13.0, -11.0, 7.0), (0.0, 0.0, 1.2), 38.0),
    "conveyor_tracking": ((-8.0, -5.0, 2.3), (0.0, -2.1, 0.8), 48.0),
    "synchronized_close": ((-1.8, -4.1, 2.0), (0.0, 0.0, 1.0), 55.0),
    "overhead_coordination": ((0.0, 0.0, 13.5), (0.0, 0.0, 0.0), 42.0),
    "assembly_packaging_detail": ((4.0, -3.1, 2.0), (5.6, 0.2, 0.9), 60.0),
    "final_hero": ((12.5, 10.5, 6.8), (0.7, 0.0, 1.0), 40.0),
}

_SHOT_END_OFFSETS = {
    "establishing": (2.0, 1.5, -0.3),
    "conveyor_tracking": (10.0, 0.4, 0.2),
    "synchronized_close": (3.2, 0.8, 0.5),
    "overhead_coordination": (0.0, 2.0, -1.0),
    "assembly_packaging_detail": (2.4, 1.0, 0.35),
    "final_hero": (-2.2, -1.5, -0.4),
}


def _look_at(obj: Any, target: tuple[float, float, float], mathutils: Any) -> None:
    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _set_camera_bezier_interpolation(animated: Any) -> None:
    for curve in _animation_fcurves(animated):
        for point in curve.keyframe_points:
            point.interpolation = "BEZIER"
            point.handle_left_type = "AUTO_CLAMPED"
            point.handle_right_type = "AUTO_CLAMPED"


def _make_camera(bpy: Any, collection: Any, shot: CameraShot, mathutils: Any) -> Any:
    location, target, lens = _SHOT_POSES[shot.name]
    data = bpy.data.cameras.new(f"Cam_{shot.name}_Data")
    data.lens = lens
    data.sensor_width = 36.0
    if shot.name in {"synchronized_close", "assembly_packaging_detail"}:
        data.dof.use_dof = True
        data.dof.aperture_fstop = 2.8
        data.dof.focus_distance = 4.2
    camera = bpy.data.objects.new(f"Cam_{shot.name}", data)
    collection.objects.link(camera)
    camera.location = location
    _look_at(camera, target, mathutils)

    camera.keyframe_insert(data_path="location", frame=shot.start_frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=shot.start_frame)

    offset = _SHOT_END_OFFSETS[shot.name]
    camera.location = tuple(location[i] + offset[i] for i in range(3))
    if shot.name == "conveyor_tracking":
        target = (5.5, -2.1, 0.8)
    _look_at(camera, target, mathutils)
    camera.keyframe_insert(data_path="location", frame=shot.end_frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=shot.end_frame)

    _set_camera_bezier_interpolation(camera)
    return camera


def build_camera_plan(scene: Any, bpy_module: Any | None = None) -> CameraPlan:
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore
    import mathutils

    bpy = bpy_module
    plan = cinematic_camera_plan()

    old = bpy.data.collections.get("FactoryCameras")
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    collection = bpy.data.collections.new("FactoryCameras")
    scene.collection.children.link(collection)

    cameras = {}
    for shot in plan.shots:
        cameras[shot.name] = _make_camera(bpy, collection, shot, mathutils)

    scene.timeline_markers.clear()
    for shot in plan.shots:
        marker = scene.timeline_markers.new(shot.name, frame=shot.start_frame)
        marker.camera = cameras[shot.name]

    scene.camera = cameras[plan.shots[0].name]
    return plan
