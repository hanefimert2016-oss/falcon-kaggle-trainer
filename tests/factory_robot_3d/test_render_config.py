from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_production_render_contract_matches_approved_cinematic_settings():
    from factory_robot_3d.blender.render_config import production_render_contract

    cfg = production_render_contract()

    assert cfg.engine == "CYCLES"
    assert (cfg.width, cfg.height) == (1920, 1080)
    assert cfg.fps == 24
    assert (cfg.frame_start, cfg.frame_end) == (1, 288)
    assert cfg.samples == 64
    assert cfg.adaptive_sampling is True
    assert cfg.denoising is True
    assert cfg.motion_blur is True
    assert cfg.view_transform == "AgX"


def test_device_selection_prefers_optix_then_cuda_and_never_cpu():
    from factory_robot_3d.blender.render_config import DeviceInfo, select_cycles_backend

    devices = [
        DeviceInfo("Tesla T4", "CUDA", True),
        DeviceInfo("Tesla T4", "OPTIX", True),
        DeviceInfo("Intel Xeon", "CPU", True),
    ]
    selected = select_cycles_backend(devices)
    assert selected.backend == "OPTIX"
    assert selected.device_names == ("Tesla T4",)

    selected = select_cycles_backend([
        DeviceInfo("Tesla T4", "CUDA", True),
        DeviceInfo("Intel Xeon", "CPU", True),
    ])
    assert selected.backend == "CUDA"

    with pytest.raises(RuntimeError, match="GPU"):
        select_cycles_backend([DeviceInfo("Intel Xeon", "CPU", True)])


def test_scene_validation_contract_requires_twenty_robots_five_cells_and_six_shots():
    from factory_robot_3d.blender.validate_scene import validate_scene_manifest

    report = validate_scene_manifest(
        robot_count=20,
        cell_count=5,
        frame_count=288,
        camera_shot_count=6,
        material_names={
            "PaintedMetal","BrushedMetal","Rubber","SafetyYellow",
            "Glass","EmissiveDisplay","WarningRed","EpoxyFloor",
        },
        lighting_ok=True,
    )
    assert report.ok is True
    assert report.errors == ()


def test_scene_validation_rejects_missing_scene_contract():
    from factory_robot_3d.blender.validate_scene import validate_scene_manifest

    report = validate_scene_manifest(
        robot_count=19,
        cell_count=5,
        frame_count=287,
        camera_shot_count=5,
        material_names={"PaintedMetal"},
        lighting_ok=False,
    )
    assert report.ok is False
    assert report.errors
