from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


BaseTransform = tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotRigSpec:
    robot_id: str
    base_transform: BaseTransform
    joint_names: tuple[str, ...]
    tool_mount_name: str
    status_light_name: str


@dataclass(frozen=True)
class RobotRigManifest:
    rigs: Mapping[str, RobotRigSpec]


@dataclass
class RobotRig:
    robot_id: str
    root: Any
    joints: tuple[Any, ...]
    tool_mount: Any
    status_light: Any
    links: tuple[Any, ...]


def robot_object_names(robot_id: str) -> dict[str, object]:
    if not robot_id:
        raise ValueError("robot_id must not be empty")
    return {
        "joints": tuple(f"Robot_{robot_id}_J{i}" for i in range(7)),
        "tool_mount": f"Robot_{robot_id}_ToolMount",
        "status_light": f"Robot_{robot_id}_StatusLight",
    }


def build_robot_manifest(
    robot_ids: Sequence[str],
    base_transforms: Sequence[BaseTransform],
) -> RobotRigManifest:
    if len(robot_ids) != 20 or len(base_transforms) != 20:
        raise ValueError("cinematic robot manifest requires exactly 20 robots")
    if len(set(robot_ids)) != 20:
        raise ValueError("cinematic robot manifest requires 20 unique robot IDs")
    normalized_bases = tuple(tuple(float(v) for v in base) for base in base_transforms)
    if len(set(normalized_bases)) != 20:
        raise ValueError("robot base transforms must be unique")

    rigs: dict[str, RobotRigSpec] = {}
    for robot_id, base in zip(robot_ids, normalized_bases):
        if len(base) != 4:
            raise ValueError("robot base transform must contain x, y, z, yaw")
        names = robot_object_names(robot_id)
        rigs[robot_id] = RobotRigSpec(
            robot_id=robot_id,
            base_transform=base,  # type: ignore[arg-type]
            joint_names=names["joints"],  # type: ignore[arg-type]
            tool_mount_name=str(names["tool_mount"]),
            status_light_name=str(names["status_light"]),
        )
    return RobotRigManifest(rigs=rigs)


def _move_to_collection(obj: Any, collection: Any) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def _new_empty(bpy: Any, name: str, collection: Any, parent: Any | None = None) -> Any:
    obj = bpy.data.objects.new(name, None)
    collection.objects.link(obj)
    obj.empty_display_type = "SPHERE"
    obj.empty_display_size = 0.08
    if parent is not None:
        obj.parent = parent
    return obj


def _add_uv_sphere(
    bpy: Any,
    name: str,
    radius: float,
    collection: Any,
    material: Any,
    parent: Any,
    location: tuple[float, float, float],
) -> Any:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius)
    obj = bpy.context.object
    obj.name = name
    obj.location = location
    obj.parent = parent
    obj.data.materials.append(material)
    _move_to_collection(obj, collection)
    bevel = obj.modifiers.new(name="JointBevel", type="BEVEL")
    bevel.width = radius * 0.05
    bevel.segments = 2
    return obj


def _add_link(
    bpy: Any,
    name: str,
    length: float,
    radius: float,
    collection: Any,
    material: Any,
    parent: Any,
    location: tuple[float, float, float],
) -> Any:
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=length)
    obj = bpy.context.object
    obj.name = name
    obj.location = location
    obj.parent = parent
    obj.data.materials.append(material)
    _move_to_collection(obj, collection)
    bevel = obj.modifiers.new(name="LinkEdge", type="BEVEL")
    bevel.width = radius * 0.08
    bevel.segments = 3
    return obj


def create_robot_rig(
    robot_id: str,
    base_transform: BaseTransform,
    *,
    materials: Mapping[str, Any] | None = None,
    bpy_module: Any | None = None,
    collection: Any | None = None,
) -> RobotRig:
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore
    bpy = bpy_module

    if materials is None:
        from .materials import ensure_materials
        materials = ensure_materials(bpy)

    if collection is None:
        collection = bpy.data.collections.get("FactoryRobots")
        if collection is None:
            collection = bpy.data.collections.new("FactoryRobots")
            bpy.context.scene.collection.children.link(collection)

    x, y, z, yaw = (float(v) for v in base_transform)
    names = robot_object_names(robot_id)
    root = _new_empty(bpy, f"Robot_{robot_id}_Root", collection)
    root.location = (x, y, z)
    root.rotation_euler = (0.0, 0.0, yaw)

    # Pivot chain; geometry is attached below each pivot so animation rotates
    # the pivot without changing mesh origins.
    offsets = (
        (0.0, 0.0, 0.18),
        (0.0, 0.0, 0.30),
        (0.0, 0.0, 0.34),
        (0.0, 0.0, 0.30),
        (0.0, 0.0, 0.24),
        (0.0, 0.0, 0.20),
        (0.0, 0.0, 0.16),
    )
    link_lengths = (0.28, 0.40, 0.38, 0.34, 0.28, 0.24, 0.18)
    joint_axes = ("Z", "Y", "Y", "X", "Y", "X", "Z")

    joints: list[Any] = []
    links: list[Any] = []
    parent = root
    for index, (joint_name, offset, length, axis) in enumerate(
        zip(names["joints"], offsets, link_lengths, joint_axes)  # type: ignore[arg-type]
    ):
        joint = _new_empty(bpy, joint_name, collection, parent=parent)
        joint.location = offset
        joint["factory_joint_axis"] = axis
        joint["factory_joint_index"] = index
        _add_uv_sphere(
            bpy,
            f"Robot_{robot_id}_JointHousing_{index}",
            0.13 if index < 4 else 0.10,
            collection,
            materials["Rubber"],
            joint,
            (0.0, 0.0, 0.0),
        )
        link = _add_link(
            bpy,
            f"Robot_{robot_id}_Link{index}",
            length,
            0.095 if index < 4 else 0.065,
            collection,
            materials["PaintedMetal"],
            joint,
            (0.0, 0.0, length * 0.5),
        )
        joints.append(joint)
        links.append(link)
        parent = joint

    tool_mount = _new_empty(bpy, str(names["tool_mount"]), collection, parent=parent)
    tool_mount.location = (0.0, 0.0, 0.18)
    _add_link(
        bpy,
        f"Robot_{robot_id}_ToolBody",
        0.18,
        0.055,
        collection,
        materials["BrushedMetal"],
        tool_mount,
        (0.0, 0.0, 0.09),
    )

    status_light = _new_empty(bpy, str(names["status_light"]), collection, parent=root)
    status_light.location = (0.18, -0.18, 0.22)
    _add_uv_sphere(
        bpy,
        f"Robot_{robot_id}_StatusLens",
        0.045,
        collection,
        materials["WarningRed"],
        status_light,
        (0.0, 0.0, 0.0),
    )

    return RobotRig(
        robot_id=robot_id,
        root=root,
        joints=tuple(joints),
        tool_mount=tool_mount,
        status_light=status_light,
        links=tuple(links),
    )
