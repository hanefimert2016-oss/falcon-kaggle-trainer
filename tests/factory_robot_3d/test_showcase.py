from __future__ import annotations

from factory_robot_3d.config import FactoryConfig
from factory_robot_3d.simulation import run_simulation


def test_full_showcase_meets_multi_robot_production_success_contract():
    result = run_simulation(FactoryConfig.cinematic_default())

    assert result.robot_count == 20
    assert result.fixed_base_robot_count == 20
    assert len(result.frame_samples) == 288
    assert result.completed_parts >= 6
    assert result.lost_parts == 0
    assert result.transfer_conflicts == 0
    assert {event.robot_id for event in result.telemetry} == {
        f"R{i:02d}" for i in range(1, 21)
    }
