from __future__ import annotations

from dataclasses import replace
import json

from factory_robot_3d.config import FactoryConfig
from factory_robot_3d.simulation import run_simulation


def test_export_contains_contiguous_frames_all_robots_and_all_telemetry_ids(tmp_path):
    from factory_robot_3d.export import write_simulation_artifacts

    config = FactoryConfig.cinematic_default()
    result = run_simulation(config)
    manifest = write_simulation_artifacts(result, tmp_path)

    animation = json.loads(manifest.animation_json.read_text())
    frames = animation["frames"]

    assert [frame["frame_index"] for frame in frames] == list(range(1, 289))
    assert all(len(frame["robots"]) == 20 for frame in frames)

    telemetry_ids = {
        row.split(",", 1)[0]
        for row in manifest.telemetry_csv.read_text().splitlines()[1:]
        if row
    }
    assert telemetry_ids == {f"R{i:02d}" for i in range(1, 21)}


def test_same_seed_writes_identical_animation_json(tmp_path):
    from factory_robot_3d.export import write_simulation_artifacts

    config = replace(FactoryConfig.cinematic_default(), duration_s=0.5)
    a = write_simulation_artifacts(run_simulation(config), tmp_path / "a")
    b = write_simulation_artifacts(run_simulation(config), tmp_path / "b")

    assert a.animation_json.read_bytes() == b.animation_json.read_bytes()
