from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .materials import ensure_materials


@dataclass(frozen=True)
class SceneManifest:
    object_groups: set[str]
    expected_counts: Mapping[str, int]
    built_counts: Mapping[str, int] = field(default_factory=dict)

    def validate_counts(self, counts: Mapping[str, int] | None = None) -> list[str]:
        actual = dict(self.built_counts if counts is None else counts)
        errors: list[str] = []
        for group in sorted(self.object_groups):
            expected = int(self.expected_counts[group])
            got = int(actual.get(group, 0))
            if got < expected:
                errors.append(f"{group}: expected at least {expected}, got {got}")
        return errors


_EXPECTED_COUNTS = {
    "epoxy_floor": 1,
    "main_conveyors": 2,
    "transfer_conveyor": 1,
    "cell_bases": 5,
    "safety_fencing": 20,
    "structural_columns": 8,
    "control_panels": 5,
    "packaging_bins": 2,
    "warning_lights": 5,
    "ceiling_fixtures": 12,
}


def required_scene_manifest() -> SceneManifest:
    return SceneManifest(
        object_groups=set(_EXPECTED_COUNTS),
        expected_counts=dict(_EXPECTED_COUNTS),
    )


def _collection(bpy: Any, scene: Any) -> Any:
    old = bpy.data.collections.get("FactoryShell")
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    collection = bpy.data.collections.new("FactoryShell")
    scene.collection.children.link(collection)
    return collection


def _move_to_collection(obj: Any, collection: Any) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def _add_box(
    bpy: Any,
    collection: Any,
    name: str,
    location: tuple[float, float, float],
    size: tuple[float, float, float],
    material: Any,
    bevel: float = 0.04,
) -> Any:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0.0:
        mod = obj.modifiers.new(name="EdgeSoftening", type="BEVEL")
        mod.width = bevel
        mod.segments = 3
    obj.data.materials.append(material)
    _move_to_collection(obj, collection)
    return obj


def build_factory_shell(scene: Any, bundle: Any, bpy_module: Any | None = None) -> SceneManifest:
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore

    if int(bundle.robot_count) != 20:
        raise ValueError("cinematic factory shell requires a 20-robot bundle")

    bpy = bpy_module
    materials = ensure_materials(bpy)
    collection = _collection(bpy, scene)
    counts = {group: 0 for group in _EXPECTED_COUNTS}

    _add_box(
        bpy, collection, "Factory_EpoxyFloor", (0.0, 0.0, -0.12),
        (20.0, 10.0, 0.24), materials["EpoxyFloor"], bevel=0.06,
    )
    counts["epoxy_floor"] += 1

    # Two long input/output conveyors and one central transfer line.
    for idx, y in enumerate((-2.3, 2.3), start=1):
        _add_box(
            bpy, collection, f"Factory_MainConveyor_{idx:02d}", (0.0, y, 0.48),
            (16.5, 0.85, 0.22), materials["BrushedMetal"], bevel=0.08,
        )
        _add_box(
            bpy, collection, f"Factory_MainBelt_{idx:02d}", (0.0, y, 0.61),
            (16.2, 0.66, 0.08), materials["Rubber"], bevel=0.025,
        )
        counts["main_conveyors"] += 1

    _add_box(
        bpy, collection, "Factory_TransferConveyor", (0.0, 0.0, 0.50),
        (12.0, 0.82, 0.24), materials["BrushedMetal"], bevel=0.08,
    )
    _add_box(
        bpy, collection, "Factory_TransferBelt", (0.0, 0.0, 0.63),
        (11.7, 0.62, 0.08), materials["Rubber"], bevel=0.025,
    )
    counts["transfer_conveyor"] += 1

    cell_centers = (-6.0, -3.0, 0.0, 3.0, 6.0)
    for index, x in enumerate(cell_centers, start=1):
        y = -0.15 if index % 2 else 0.15
        _add_box(
            bpy, collection, f"Factory_CellBase_C{index}", (x, y, 0.14),
            (2.45, 3.45, 0.28), materials["PaintedMetal"], bevel=0.10,
        )
        counts["cell_bases"] += 1

        panel = _add_box(
            bpy, collection, f"Factory_ControlPanel_C{index}", (x + 0.9, y - 1.55, 1.1),
            (0.34, 0.24, 1.4), materials["BrushedMetal"], bevel=0.05,
        )
        _add_box(
            bpy, collection, f"Factory_ControlScreen_C{index}", (x + 0.9, y - 1.685, 1.28),
            (0.24, 0.025, 0.46), materials["EmissiveDisplay"], bevel=0.015,
        )
        counts["control_panels"] += 1

        _add_box(
            bpy, collection, f"Factory_WarningBeacon_C{index}", (x, y - 1.56, 1.38),
            (0.16, 0.16, 0.24), materials["WarningRed"], bevel=0.04,
        )
        counts["warning_lights"] += 1

    # Safety fence perimeter: 20 posts with transparent infill panels.
    fence_specs: list[tuple[float, float]] = []
    for x in (-8.6, -6.9, -5.2, -3.5, -1.8, -0.1, 1.6, 3.3, 5.0, 6.7):
        fence_specs.append((x, -4.25))
        fence_specs.append((x, 4.25))
    for idx, (x, y) in enumerate(fence_specs, start=1):
        _add_box(
            bpy, collection, f"Factory_SafetyPost_{idx:02d}", (x, y, 1.0),
            (0.10, 0.10, 2.0), materials["SafetyYellow"], bevel=0.025,
        )
        counts["safety_fencing"] += 1

    # Structural columns frame the space and give the camera scale cues.
    columns = (
        (-8.8, -4.6), (-8.8, 4.6), (-3.0, -4.6), (-3.0, 4.6),
        (3.0, -4.6), (3.0, 4.6), (8.8, -4.6), (8.8, 4.6),
    )
    for idx, (x, y) in enumerate(columns, start=1):
        _add_box(
            bpy, collection, f"Factory_Column_{idx:02d}", (x, y, 2.8),
            (0.32, 0.32, 5.6), materials["BrushedMetal"], bevel=0.045,
        )
        counts["structural_columns"] += 1

    for idx, x in enumerate((6.7, 7.7), start=1):
        _add_box(
            bpy, collection, f"Factory_PackagingBin_{idx:02d}", (x, 1.15, 0.45),
            (0.9, 0.9, 0.9), materials["SafetyYellow"], bevel=0.08,
        )
        counts["packaging_bins"] += 1

    # Twelve visible ceiling practicals; real area lights are added by lighting.py.
    for idx, x in enumerate((-7.5, -4.5, -1.5, 1.5, 4.5, 7.5), start=1):
        for row, y in enumerate((-2.25, 2.25), start=1):
            _add_box(
                bpy, collection, f"Factory_CeilingFixture_{idx:02d}_{row}", (x, y, 5.0),
                (1.65, 0.20, 0.10), materials["Glass"], bevel=0.025,
            )
            counts["ceiling_fixtures"] += 1

    manifest = SceneManifest(
        object_groups=set(_EXPECTED_COUNTS),
        expected_counts=dict(_EXPECTED_COUNTS),
        built_counts=counts,
    )
    errors = manifest.validate_counts()
    if errors:
        raise RuntimeError("factory shell validation failed: " + "; ".join(errors))
    return manifest
