from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AnimationManifest:
    frame_count: int
    robot_ids: tuple[str, ...]
    joint_keyframes_per_robot: int
    workpiece_ids: tuple[str, ...]


def animation_manifest_from_bundle(bundle: Any) -> AnimationManifest:
    robot_ids = tuple(bundle.robot_ids)
    if len(robot_ids) != 20 or len(set(robot_ids)) != 20:
        raise ValueError("animation requires exactly 20 unique robot IDs")
    frames = tuple(bundle.frames)
    if len(frames) != 288:
        raise ValueError("animation requires exactly 288 frames")

    workpiece_ids: set[str] = set()
    for expected_index, frame in enumerate(frames, start=1):
        if int(frame.get("frame_index", -1)) != expected_index:
            raise ValueError("animation frame indices must be contiguous")
        robots = frame.get("robots")
        if not isinstance(robots, dict) or set(robots) != set(robot_ids):
            raise ValueError(f"animation frame {expected_index} is missing robot keyframes")
        for robot_id in robot_ids:
            joints = robots[robot_id]
            if not isinstance(joints, list) or len(joints) != 7:
                raise ValueError(
                    f"robot {robot_id} frame {expected_index} must contain seven joints"
                )
        workpieces.update(frame.get("workpieces", {}).keys())

    return AnimationManifest(
        frame_count=288,
        robot_ids=robot_ids,
        joint_keyframes_per_robot=288,
        workpiece_ids=tuple(sorted(workpiece_ids)),
    )


def _set_joint_rotation(joint: Any, axis: str, angle: float) -> None:
    axis = str(axis).upper()
    if axis == "X":
        joint.rotation_euler[0] = angle
    elif axis == "Y":
        joint.rotation_euler[1] = angle
    elif axis == "Z":
        joint.rotation_euler[2] = angle
    else:
        raise ValueError(f"unsupported joint axis: {axis}")


def _ensure_workpiece_object(
    bpy: Any,
    collection: Any,
    workpiece_id: str,
    material: Any | None = None,
) -> Any:
    name = f"Workpiece_{workpiece_id}"
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj

    bpy.ops.mesh.primitive_cube_add(size=0.22)
    obj = bpy.context.object
    obj.name = name
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)
    bevel = obj.modifiers.new(name="WorkpieceEdges", type="BEVEL")
    bevel.width = 0.025
    bevel.segments = 2
    if material is not None:
        obj.data.materials.append(material)
    return obj


def apply_animation(
    bundle: Any,
    rigs: Mapping[str, Any],
    workpieces: Mapping[str, Any] | None = None,
    *,
    bpy_module: Any | None = None,
    materials: Mapping[str, Any] | None = None,
) -> AnimationManifest:
    manifest = animation_manifest_from_bundle(bundle)
    if set(rigs) != set(manifest.robot_ids):
        raise ValueError("rig map must contain exactly the bundle's 20 robots")

    if bpy_module is None:
        import bpy as bpy_module  # type: ignore
    bpy = bpy_module

    if workpieces is None:
        collection = bpy.data.collections.get("FactoryWorkpieces")
        if collection is None:
            collection = bpy.data.collections.new("FactoryWorkpieces")
            bpy.context.scene.collection.children.link(collection)
        mutable_workpieces: dict[str, Any] = {}
        for workpiece_id in manifest.workpiece_ids:
            material = None if materials is None else materials.get("BrushedMetal")
            mutable_workpieces[workpiece_id] = _ensure_workpiece_object(
                bpy, collection, workpiece_id, material
            )
        workpieces = mutable_workpieces

    for frame in bundle.frames:
        frame_index = int(frame["frame_index"])

        for robot_id in manifest.robot_ids:
            rig = rigs[robot_id]
            angles = frame["robots"][robot_id]
            if len(rig.joints) != 7:
                raise ValueError(f"rig {robot_id} does not expose seven joints")
            for joint, angle in zip(rig.joints, angles):
                axis = joint.get("factory_joint_axis", "Z")
                _set_joint_rotation(joint, axis, float(angle))
                joint.keyframe_insert(data_path="rotation_euler", frame=frame_index)

        frame_workpieces = frame.get("workpieces", {})
        for workpiece_id, obj in workpieces.items():
            state = frame_workpieces.get(workpiece_id)
            if state is None:
                obj.hide_render = True
                obj.hide_viewport = True
            else:
                position = state.get("position", (0.0, 0.0, 0.0))
                obj.location = tuple(float(v) for v in position)
                obj.hide_render = False
                obj.hide_viewport = False
            obj.keyframe_insert(data_path="location", frame=frame_index)
            obj.keyframe_insert(data_path="hide_render", frame=frame_index)
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_index)

    # Simulation data should interpolate linearly between sampled states.
    for rig in rigs.values():
        for joint in rig.joints:
            action = getattr(getattr(joint, "animation_data", None), "action", None)
            if action is None:
                continue
            for fcurve in action.fcurves:
                for point in fcurve.keyframe_points:
                    point.interpolation = "LINEAR"

    for obj in workpieces.values():
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is not None:
            for fcurve in action.fcurves:
                for point in fcurve.keyframe_points:
                    point.interpolation = "LINEAR"

    return manifest
