from __future__ import annotations

import pytest


def _robot_ids():
    return tuple(f"R{i:02d}" for i in range(1, 21))


def _bases():
    return tuple((float(i), float(i % 4), 0.0, 0.0) for i in range(20))


def test_robot_manifest_contains_twenty_complete_seven_dof_rigs():
    from factory_robot_3d.blender.robot_rig import build_robot_manifest

    manifest = build_robot_manifest(_robot_ids(), _bases())

    assert len(manifest.rigs) == 20
    assert set(manifest.rigs) == set(_robot_ids())
    for robot_id, rig in manifest.rigs.items():
        assert rig.joint_names == tuple(
            f"Robot_{robot_id}_J{i}" for i in range(7)
        )
        assert rig.tool_mount_name == f"Robot_{robot_id}_ToolMount"
        assert rig.status_light_name == f"Robot_{robot_id}_StatusLight"


def test_robot_manifest_requires_unique_base_transforms():
    from factory_robot_3d.blender.robot_rig import build_robot_manifest

    bases = list(_bases())
    bases[-1] = bases[0]

    with pytest.raises(ValueError, match="base"):
        build_robot_manifest(_robot_ids(), tuple(bases))


def test_robot_manifest_requires_exactly_twenty_unique_ids():
    from factory_robot_3d.blender.robot_rig import build_robot_manifest

    with pytest.raises(ValueError, match="20"):
        build_robot_manifest(_robot_ids()[:-1], _bases()[:-1])


def test_stable_robot_object_naming_contract():
    from factory_robot_3d.blender.robot_rig import robot_object_names

    names = robot_object_names("R07")

    assert names["joints"] == tuple(f"Robot_R07_J{i}" for i in range(7))
    assert names["tool_mount"] == "Robot_R07_ToolMount"
    assert names["status_light"] == "Robot_R07_StatusLight"
