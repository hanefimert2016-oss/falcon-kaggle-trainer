from __future__ import annotations

import pytest


def test_cinematic_default_has_approved_factory_contract():
    from factory_robot_3d.config import FactoryConfig

    config = FactoryConfig.cinematic_default()

    assert config.robot_count == 20
    assert config.cell_count == 5
    assert config.fps == 24
    assert config.duration_s == 12.0
    assert config.frame_count == 288
    assert config.seed == 20261002
    assert config.role_counts == {
        "pick_place": 8,
        "assembly": 4,
        "joining": 4,
        "quality_control": 2,
        "packaging": 2,
    }
    config.validate()


def test_factory_config_rejects_role_total_that_does_not_match_robot_count():
    from factory_robot_3d.config import FactoryConfig

    config = FactoryConfig(
        robot_count=20,
        cell_count=5,
        fps=24,
        duration_s=12.0,
        seed=20261002,
        role_counts={
            "pick_place": 8,
            "assembly": 4,
            "joining": 4,
            "quality_control": 2,
            "packaging": 1,
        },
    )

    with pytest.raises(ValueError, match="role counts"):
        config.validate()


def test_domain_state_contracts_expose_explicit_owners():
    from factory_robot_3d.models import (
        RobotSpec,
        TaskState,
        TransferZoneState,
        WorkpieceState,
    )

    robot = RobotSpec(
        robot_id="R01",
        cell_id="C1",
        role="pick_place",
        base_position=(1.0, 2.0, 0.0),
        base_yaw=0.5,
    )
    workpiece = WorkpieceState(workpiece_id="P001", stage="input")
    task = TaskState(task_id="T001", stage="pick_place", workpiece_id="P001")
    zone = TransferZoneState(zone_id="Z01")

    assert robot.robot_id == "R01"
    assert robot.base_position == (1.0, 2.0, 0.0)
    assert workpiece.owner_robot_id is None
    assert task.owner_robot_id is None
    assert zone.owner_robot_id is None

    workpiece.owner_robot_id = "R01"
    task.owner_robot_id = "R01"
    zone.owner_robot_id = "R01"

    assert workpiece.owner_robot_id == "R01"
    assert task.owner_robot_id == "R01"
    assert zone.owner_robot_id == "R01"
