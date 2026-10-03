from __future__ import annotations

import pytest

from factory_robot_3d.config import FactoryConfig
from factory_robot_3d.layout import build_factory_layout
from factory_robot_3d.models import TaskState, TransferZoneState, WorkpieceState


def make_scheduler_and_world():
    from factory_robot_3d.scheduler import FactoryScheduler, FactoryWorldState

    layout = build_factory_layout(FactoryConfig.cinematic_default())
    world = FactoryWorldState(
        workpieces={
            "P001": WorkpieceState("P001", "pick_place"),
            "P002": WorkpieceState("P002", "pick_place"),
        },
        tasks={
            "T002": TaskState("T002", "pick_place", "P002"),
            "T001": TaskState("T001", "pick_place", "P001"),
        },
        zones={"Z_PICK_ASSEMBLY": TransferZoneState("Z_PICK_ASSEMBLY")},
    )
    return FactoryScheduler(layout, seed=20261002), world


def test_workpiece_has_exactly_one_owner_and_can_be_reacquired_after_release():
    scheduler, _ = make_scheduler_and_world()

    assert scheduler.acquire_workpiece("P001", "R01") is True
    assert scheduler.acquire_workpiece("P001", "R02") is False
    assert scheduler.release_workpiece("P001", "R01") is True
    assert scheduler.acquire_workpiece("P001", "R02") is True


def test_transfer_zone_serializes_competing_robots():
    scheduler, _ = make_scheduler_and_world()

    assert scheduler.acquire_zone("Z_PICK_ASSEMBLY", "R01") is True
    assert scheduler.acquire_zone("Z_PICK_ASSEMBLY", "R02") is False
    assert scheduler.release_zone("Z_PICK_ASSEMBLY", "R01") is True
    assert scheduler.acquire_zone("Z_PICK_ASSEMBLY", "R02") is True


def test_scheduler_order_is_stable_for_equal_ready_tasks():
    scheduler, world = make_scheduler_and_world()

    commands = scheduler.step(0.0, world)

    assert [(c.task_id, c.robot_id) for c in commands] == [
        ("T001", "R01"),
        ("T002", "R02"),
    ]


def test_workpiece_cannot_skip_production_stage():
    scheduler, world = make_scheduler_and_world()

    scheduler.advance_workpiece_stage(world.workpieces["P001"], "assembly")
    assert world.workpieces["P001"].stage == "assembly"

    with pytest.raises(ValueError, match="skip"):
        scheduler.advance_workpiece_stage(world.workpieces["P001"], "quality_control")
