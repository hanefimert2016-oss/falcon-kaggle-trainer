from __future__ import annotations


def test_factory_scene_manifest_declares_all_required_groups():
    from factory_robot_3d.blender.factory_scene import required_scene_manifest

    manifest = required_scene_manifest()

    assert manifest.object_groups == {
        "epoxy_floor",
        "main_conveyors",
        "transfer_conveyor",
        "cell_bases",
        "safety_fencing",
        "structural_columns",
        "control_panels",
        "packaging_bins",
        "warning_lights",
        "ceiling_fixtures",
    }
    assert manifest.expected_counts["main_conveyors"] == 2
    assert manifest.expected_counts["transfer_conveyor"] == 1
    assert manifest.expected_counts["cell_bases"] == 5


def test_material_recipes_cover_cinematic_factory_look():
    from factory_robot_3d.blender.materials import MATERIAL_RECIPES

    assert set(MATERIAL_RECIPES) == {
        "PaintedMetal",
        "BrushedMetal",
        "Rubber",
        "SafetyYellow",
        "Glass",
        "EmissiveDisplay",
        "WarningRed",
        "EpoxyFloor",
    }

    assert MATERIAL_RECIPES["PaintedMetal"].metallic > 0.5
    assert 0.15 <= MATERIAL_RECIPES["BrushedMetal"].roughness <= 0.5
    assert MATERIAL_RECIPES["EmissiveDisplay"].emission_strength > 1.0
    assert MATERIAL_RECIPES["Glass"].transmission > 0.8
    assert MATERIAL_RECIPES["EpoxyFloor"].roughness < 0.45


def test_scene_manifest_validation_rejects_missing_group():
    from factory_robot_3d.blender.factory_scene import required_scene_manifest

    manifest = required_scene_manifest()
    counts = dict(manifest.expected_counts)
    counts["safety_fencing"] = 0

    errors = manifest.validate_counts(counts)

    assert any("safety_fencing" in error for error in errors)
