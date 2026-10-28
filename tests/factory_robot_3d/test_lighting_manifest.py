from __future__ import annotations

import pytest


def test_lighting_manifest_meets_cinematic_factory_requirements():
    from factory_robot_3d.blender.lighting import factory_lighting_manifest

    manifest = factory_lighting_manifest()

    assert manifest.ceiling_area_lights >= 12
    assert manifest.practical_emissive_fixtures >= 12
    assert manifest.hero_area_lights >= 2
    assert manifest.warning_beacons == 5
    assert manifest.color_management == "AgX"
    assert 0.0 < manifest.volume_density <= 0.02
    assert -2.0 <= manifest.exposure <= 2.0


def test_lighting_manifest_rejects_obscuring_volumetrics():
    from factory_robot_3d.blender.lighting import LightingManifest

    with pytest.raises(ValueError, match="volume"):
        LightingManifest(
            ceiling_area_lights=12,
            practical_emissive_fixtures=12,
            hero_area_lights=2,
            warning_beacons=5,
            volume_density=0.08,
            exposure=0.0,
            color_management="AgX",
        ).validate()


def test_lighting_manifest_requires_agx_and_practical_sources():
    from factory_robot_3d.blender.lighting import LightingManifest

    manifest = LightingManifest(
        ceiling_area_lights=0,
        practical_emissive_fixtures=0,
        hero_area_lights=0,
        warning_beacons=0,
        volume_density=0.0,
        exposure=0.0,
        color_management="Standard",
    )
    with pytest.raises(ValueError):
        manifest.validate()
