from __future__ import annotations

from dataclasses import replace

import pytest

from factory_robot_3d.config import FactoryConfig


def test_one_second_smoke_creates_twenty_fixed_robots_and_24_frames():
    from factory_robot_3d.simulation import run_simulation

    config = replace(FactoryConfig.cinematic_default(), duration_s=1.0)
    result = run_simulation(config)

    assert result.robot_count == 20
    assert result.fixed_base_robot_count == 20
    assert len(result.frame_samples) == 24
    assert {event.robot_id for event in result.telemetry} == {
        f"R{i:02d}" for i in range(1, 21)
    }


def test_unreachable_target_fails_without_mutating_another_robot():
    from factory_robot_3d.simulation import FactorySimulation

    config = replace(FactoryConfig.cinematic_default(), duration_s=0.25)
    sim = FactorySimulation(config)
    try:
        before = sim.get_joint_positions("R02")
        accepted = sim.command_robot_target("R01", (100.0, 100.0, 100.0))
        after = sim.get_joint_positions("R02")

        assert accepted is False
        assert after == pytest.approx(before, abs=1e-12)
        assert sim.telemetry[-1].robot_id == "R01"
        assert sim.telemetry[-1].result == "failed_unreachable"
    finally:
        sim.close()
