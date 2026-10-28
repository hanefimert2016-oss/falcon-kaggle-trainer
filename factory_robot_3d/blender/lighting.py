from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LightingManifest:
    ceiling_area_lights: int
    practical_emissive_fixtures: int
    hero_area_lights: int
    warning_beacons: int
    volume_density: float
    exposure: float
    color_management: str

    def validate(self) -> None:
        errors: list[str] = []
        if self.ceiling_area_lights < 12:
            errors.append("ceiling area-light rows require at least 12 sources")
        if self.practical_emissive_fixtures < 12:
            errors.append("practical emissive fixtures require at least 12 sources")
        if self.hero_area_lights < 2:
            errors.append("hero key/fill requires at least 2 area lights")
        if self.warning_beacons < 5:
            errors.append("warning beacons require 5 sources")
        if not (0.0 < self.volume_density <= 0.02):
            errors.append("volume density must be > 0 and <= 0.02")
        if not (-2.0 <= self.exposure <= 2.0):
            errors.append("exposure must stay within cinematic safety range")
        if self.color_management != "AgX":
            errors.append("color management must use AgX")
        if errors:
            raise ValueError("; ".join(errors))


def factory_lighting_manifest() -> LightingManifest:
    manifest = LightingManifest(
        ceiling_area_lights=12,
        practical_emissive_fixtures=12,
        hero_area_lights=4,
        warning_beacons=5,
        volume_density=0.008,
        exposure=0.35,
        color_management="AgX",
    )
    manifest.validate()
    return manifest


def _lighting_collection(bpy: Any, scene: Any) -> Any:
    old = bpy.data.collections.get("FactoryLighting")
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    collection = bpy.data.collections.new("FactoryLighting")
    scene.collection.children.link(collection)
    return collection


def _area_light(
    bpy: Any,
    collection: Any,
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float,
    color: tuple[float, float, float],
) -> Any:
    data = bpy.data.lights.new(name=f"{name}_Data", type="AREA")
    data.energy = energy
    data.shape = "RECTANGLE"
    data.size = size
    data.size_y = max(0.3, size * 0.18)
    data.color = color
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    obj.rotation_euler = (0.0, 0.0, 0.0)
    collection.objects.link(obj)
    return obj


def _point_light(
    bpy: Any,
    collection: Any,
    name: str,
    location: tuple[float, float, float],
    energy: float,
    color: tuple[float, float, float],
) -> Any:
    data = bpy.data.lights.new(name=f"{name}_Data", type="POINT")
    data.energy = energy
    data.color = color
    data.shadow_soft_size = 0.18
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    collection.objects.link(obj)
    return obj


def configure_factory_lighting(scene: Any, bpy_module: Any | None = None) -> LightingManifest:
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore
    bpy = bpy_module
    manifest = factory_lighting_manifest()
    collection = _lighting_collection(bpy, scene)

    ceiling_positions = (
        (-7.5, -2.0, 4.65), (-4.5, -2.0, 4.65), (-1.5, -2.0, 4.65),
        (1.5, -2.0, 4.65), (4.5, -2.0, 4.65), (7.5, -2.0, 4.65),
        (-7.5, 2.0, 4.65), (-4.5, 2.0, 4.65), (-1.5, 2.0, 4.65),
        (1.5, 2.0, 4.65), (4.5, 2.0, 4.65), (7.5, 2.0, 4.65),
    )
    for index, pos in enumerate(ceiling_positions, start=1):
        _area_light(
            bpy,
            collection,
            f"Factory_CeilingArea_{index:02d}",
            pos,
            energy=780.0,
            size=2.0,
            color=(0.88, 0.94, 1.0),
        )

    hero_specs = (
        ("HeroKey_Left", (-5.5, -3.6, 3.5), 1200.0, (1.0, 0.82, 0.66)),
        ("HeroKey_Right", (5.7, -3.2, 3.2), 1050.0, (0.70, 0.84, 1.0)),
        ("HeroFill_Left", (-2.2, 3.5, 2.8), 700.0, (0.72, 0.86, 1.0)),
        ("HeroFill_Right", (3.5, 3.6, 2.6), 650.0, (1.0, 0.78, 0.60)),
    )
    for name, pos, energy, color in hero_specs:
        _area_light(bpy, collection, f"Factory_{name}", pos, energy, 3.0, color)

    cell_x = (-6.0, -3.0, 0.0, 3.0, 6.0)
    for index, x in enumerate(cell_x, start=1):
        color = (1.0, 0.035, 0.015) if index % 2 else (1.0, 0.36, 0.015)
        _point_light(
            bpy,
            collection,
            f"Factory_BeaconLight_C{index}",
            (x, -1.55, 1.65),
            energy=130.0,
            color=color,
        )

    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = manifest.exposure
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    world = scene.world or bpy.data.worlds.new("FactoryWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    background = nodes.get("Background")
    output = nodes.get("World Output")
    if background is not None:
        background.inputs["Color"].default_value = (0.012, 0.018, 0.028, 1.0)
        background.inputs["Strength"].default_value = 0.18
    volume = nodes.get("FactoryAtmosphere")
    if volume is None:
        volume = nodes.new("ShaderNodeVolumeScatter")
        volume.name = "FactoryAtmosphere"
    volume.inputs["Color"].default_value = (0.62, 0.72, 0.82, 1.0)
    volume.inputs["Density"].default_value = manifest.volume_density
    volume.inputs["Anisotropy"].default_value = 0.18
    if output is not None:
        for link in list(output.inputs["Volume"].links):
            links.remove(link)
        links.new(volume.outputs["Volume"], output.inputs["Volume"])

    return manifest
