from __future__ import annotations

from collections import Counter

from factory_robot_3d.config import FactoryConfig


def test_build_factory_layout_has_approved_robot_and_cell_counts():
    from factory_robot_3d.layout import build_factory_layout

    layout = build_factory_layout(FactoryConfig.cinematic_default())

    assert len(layout.robots) == 20
    assert len({robot.robot_id for robot in layout.robots}) == 20
    assert len(layout.cells) == 5

    robots_per_cell = Counter(robot.cell_id for robot in layout.robots)
    assert robots_per_cell == {f"C{i}": 4 for i in range(1, 6)}

    roles = Counter(robot.role for robot in layout.robots)
    assert roles == {
        "pick_place": 8,
        "assembly": 4,
        "joining": 4,
        "quality_control": 2,
        "packaging": 2,
    }


def test_robot_base_positions_are_unique():
    from factory_robot_3d.layout import build_factory_layout

    layout = build_factory_layout(FactoryConfig.cinematic_default())
    positions = [robot.base_position for robot in layout.robots]

    assert len(set(positions)) == len(positions)


def test_factory_layout_contains_two_main_conveyors_and_one_transfer_line():
    from factory_robot_3d.layout import build_factory_layout

    layout = build_factory_layout(FactoryConfig.cinematic_default())

    assert len(layout.main_conveyors) == 2
    assert layout.transfer_conveyor.conveyor_id == "TRANSFER"
    assert {c.conveyor_id for c in layout.main_conveyors} == {"MAIN_IN", "MAIN_OUT"}


def test_transfer_zones_cover_each_adjacent_production_stage():
    from factory_robot_3d.layout import build_factory_layout

    layout = build_factory_layout(FactoryConfig.cinematic_default())

    pairs = {(zone.from_stage, zone.to_stage) for zone in layout.transfer_zones}

    assert pairs == {
        ("pick_place", "assembly"),
        ("assembly", "joining"),
        ("joining", "quality_control"),
        ("quality_control", "packaging"),
    }
