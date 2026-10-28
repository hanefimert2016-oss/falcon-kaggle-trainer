from __future__ import annotations

import json
import math

import pytest


def _write_bundle(tmp_path, *, frame_count=288, robot_count=20, duplicate_frame=False, missing_robot=False, nonfinite=False):
    config = {
        "robot_count": robot_count,
        "cell_count": 5,
        "fps": 24,
        "duration_s": 12.0,
        "frame_count": frame_count,
        "seed": 20261002,
        "role_counts": {
            "pick_place": 8,
            "assembly": 4,
            "joining": 4,
            "quality_control": 2,
            "packaging": 2,
        },
    }
    frames = []
    for idx in range(1, frame_count + 1):
        frame_index = 1 if duplicate_frame and idx == 2 else idx
        robots = {
            f"R{i:02d}": [0.0] * 7
            for i in range(1, robot_count + 1)
        }
        if missing_robot and idx == 8:
            robots.pop("R20", None)
        if nonfinite and idx == 9:
            robots["R01"][3] = float("nan")
        frames.append({
            "frame_index": frame_index,
            "time_s": idx / 24.0,
            "robots": robots,
            "workpieces": {},
            "conveyor_offsets": {},
            "warning_lights": {},
            "active_task_ids": [],
        })
    (tmp_path / "factory_config.json").write_text(json.dumps(config))
    (tmp_path / "animation.json").write_text(json.dumps({"schema_version": 1, "frames": frames}, allow_nan=True))
    (tmp_path / "summary.json").write_text(json.dumps({
        "success": True,
        "robot_count": robot_count,
        "cell_count": 5,
        "completed_parts": 8,
        "lost_parts": 0,
        "transfer_conflicts": 0,
        "simulation_frames": frame_count,
        "fps": 24,
        "duration_s": 12.0,
        "seed": 20261002,
    }))


def test_load_animation_bundle_accepts_approved_contract(tmp_path):
    from factory_robot_3d.blender.schema import load_animation_bundle

    _write_bundle(tmp_path)
    bundle = load_animation_bundle(tmp_path)

    assert bundle.robot_count == 20
    assert len(bundle.frames) == 288
    assert bundle.frame_indices == tuple(range(1, 289))


@pytest.mark.parametrize("frame_count", [287, 289])
def test_bundle_rejects_wrong_frame_count(tmp_path, frame_count):
    from factory_robot_3d.blender.schema import load_animation_bundle

    _write_bundle(tmp_path, frame_count=frame_count)
    with pytest.raises(ValueError, match="288"):
        load_animation_bundle(tmp_path)


def test_bundle_rejects_duplicate_frame_index(tmp_path):
    from factory_robot_3d.blender.schema import load_animation_bundle

    _write_bundle(tmp_path, duplicate_frame=True)
    with pytest.raises(ValueError, match="frame"):
        load_animation_bundle(tmp_path)


def test_bundle_rejects_missing_robot_in_any_frame(tmp_path):
    from factory_robot_3d.blender.schema import load_animation_bundle

    _write_bundle(tmp_path, missing_robot=True)
    with pytest.raises(ValueError, match="robot"):
        load_animation_bundle(tmp_path)


def test_bundle_rejects_robot_count_other_than_twenty(tmp_path):
    from factory_robot_3d.blender.schema import load_animation_bundle

    _write_bundle(tmp_path, robot_count=19)
    with pytest.raises(ValueError, match="20"):
        load_animation_bundle(tmp_path)


def test_bundle_rejects_nonfinite_joint_values(tmp_path):
    from factory_robot_3d.blender.schema import load_animation_bundle

    _write_bundle(tmp_path, nonfinite=True)
    with pytest.raises(ValueError, match="finite"):
        load_animation_bundle(tmp_path)


def test_bundle_requires_simulation_summary_fields(tmp_path):
    from factory_robot_3d.blender.schema import load_animation_bundle

    _write_bundle(tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"success": True}))
    with pytest.raises(ValueError, match="summary"):
        load_animation_bundle(tmp_path)
