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


def test_showcase_enforces_unique_workpiece_ownership_and_records_transfers():
    result = run_simulation(FactoryConfig.cinematic_default())

    for frame in result.frame_samples:
        owners = [
            state.get("owner_robot_id")
            for state in frame.workpieces.values()
            if state.get("owner_robot_id")
        ]
        assert len(owners) == len(set(owners))

    transfer_events = [
        event for event in result.telemetry
        if event.workpiece_id and event.transfer_zone_id
    ]
    assert transfer_events
    assert {event.transfer_zone_id for event in transfer_events} == {
        "Z_PICK_ASSEMBLY",
        "Z_ASSEMBLY_JOINING",
        "Z_JOINING_QC",
        "Z_QC_PACKAGING",
    }
    assert result.transfer_conflicts == 0
