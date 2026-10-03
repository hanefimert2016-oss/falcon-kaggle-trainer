from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class MaterialRecipe:
    base_color: Color
    metallic: float = 0.0
    roughness: float = 0.5
    transmission: float = 0.0
    alpha: float = 1.0
    emission_color: Color | None = None
    emission_strength: float = 0.0


MATERIAL_RECIPES: dict[str, MaterialRecipe] = {
    "PaintedMetal": MaterialRecipe((0.92, 0.18, 0.025, 1.0), metallic=0.72, roughness=0.28),
    "BrushedMetal": MaterialRecipe((0.34, 0.38, 0.42, 1.0), metallic=0.95, roughness=0.32),
    "Rubber": MaterialRecipe((0.018, 0.022, 0.026, 1.0), metallic=0.0, roughness=0.72),
    "SafetyYellow": MaterialRecipe((0.96, 0.55, 0.02, 1.0), metallic=0.28, roughness=0.34),
    "Glass": MaterialRecipe((0.22, 0.35, 0.42, 0.22), roughness=0.08, transmission=0.95, alpha=0.22),
    "EmissiveDisplay": MaterialRecipe(
        (0.015, 0.055, 0.07, 1.0),
        roughness=0.25,
        emission_color=(0.04, 0.72, 1.0, 1.0),
        emission_strength=5.5,
    ),
    "WarningRed": MaterialRecipe(
        (0.25, 0.01, 0.005, 1.0),
        roughness=0.24,
        emission_color=(1.0, 0.025, 0.005, 1.0),
        emission_strength=8.0,
    ),
    "EpoxyFloor": MaterialRecipe((0.075, 0.085, 0.10, 1.0), metallic=0.22, roughness=0.30),
}


def _set_input(node: Any, name: str, value: Any) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def ensure_materials(bpy_module: Any | None = None) -> dict[str, Any]:
    """Create/reuse all project-owned Principled BSDF materials.

    bpy is imported lazily so manifest/schema tests remain runnable on normal
    Python/GitHub Actions hosts without Blender installed.
    """
    if bpy_module is None:
        import bpy as bpy_module  # type: ignore

    built: dict[str, Any] = {}
    for name, recipe in MATERIAL_RECIPES.items():
        material = bpy_module.data.materials.get(name)
        if material is None:
            material = bpy_module.data.materials.new(name=name)
        material.use_nodes = True
        material.diffuse_color = recipe.base_color
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"

        tree = material.node_tree
        principled = tree.nodes.get("Principled BSDF")
        if principled is None:
            tree.nodes.clear()
            principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        _set_input(principled, "Base Color", recipe.base_color)
        _set_input(principled, "Metallic", recipe.metallic)
        _set_input(principled, "Roughness", recipe.roughness)
        _set_input(principled, "Alpha", recipe.alpha)
        _set_input(principled, "Transmission Weight", recipe.transmission)
        _set_input(principled, "Transmission", recipe.transmission)
        if recipe.emission_color is not None:
            _set_input(principled, "Emission Color", recipe.emission_color)
            _set_input(principled, "Emission", recipe.emission_color)
            _set_input(principled, "Emission Strength", recipe.emission_strength)

        built[name] = material
    return built
