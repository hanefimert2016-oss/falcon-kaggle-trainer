from __future__ import annotations

import json


def test_probe_payload_prefers_optix_t4():
    from factory_robot_3d.blender.probe_gpu import probe_payload
    from factory_robot_3d.blender.render_config import DeviceInfo

    payload = probe_payload([
        DeviceInfo("Tesla T4", "CUDA", True),
        DeviceInfo("Tesla T4", "OPTIX", True),
        DeviceInfo("CPU", "CPU", True),
    ], blender_version="5.2.2")

    assert payload["backend"] == "OPTIX"
    assert payload["device_names"] == ["Tesla T4"]
    assert payload["blender_version"] == "5.2.2"


def test_probe_payload_rejects_non_t4_gpu():
    import pytest
    from factory_robot_3d.blender.probe_gpu import probe_payload
    from factory_robot_3d.blender.render_config import DeviceInfo

    with pytest.raises(RuntimeError, match="T4"):
        probe_payload([DeviceInfo("RTX 4090", "CUDA", True)], blender_version="5.2.2")
