from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_cinematic_camera_plan_covers_all_288_frames_without_gaps():
    from factory_robot_3d.blender.cameras import cinematic_camera_plan

    plan = cinematic_camera_plan()
    plan.validate()

    assert [(shot.start_frame, shot.end_frame) for shot in plan.shots] == [
        (1, 48),
        (49, 96),
        (97, 144),
        (145, 192),
        (193, 240),
        (241, 288),
    ]
    assert [shot.name for shot in plan.shots] == [
        "establishing",
        "conveyor_tracking",
        "synchronized_close",
        "overhead_coordination",
        "assembly_packaging_detail",
        "final_hero",
    ]


def test_camera_plan_validation_rejects_gap_or_overlap():
    from factory_robot_3d.blender.cameras import CameraPlan, CameraShot

    shots = (
        CameraShot("a", 1, 48),
        CameraShot("b", 50, 96),
    )
    with pytest.raises(ValueError, match="contiguous"):
        CameraPlan(shots=shots, total_frames=96).validate()


def test_animation_manifest_requires_twenty_robots_and_288_frames():
    from factory_robot_3d.blender.animation import animation_manifest_from_bundle

    robot_ids = tuple(f"R{i:02d}" for i in range(1, 21))
    frames = tuple(
        {
            "frame_index": frame,
            "robots": {robot_id: [0.0] * 7 for robot_id in robot_ids},
            "workpieces": {},
        }
        for frame in range(1, 289)
    )
    bundle = SimpleNamespace(robot_ids=robot_ids, frames=frames)

    manifest = animation_manifest_from_bundle(bundle)

    assert manifest.frame_count == 288
    assert manifest.robot_ids == robot_ids
    assert manifest.joint_keyframes_per_robot == 288


def test_animation_manifest_rejects_missing_robot_keyframe():
    from factory_robot_3d.blender.animation import animation_manifest_from_bundle

    robot_ids = tuple(f"R{i:02d}" for i in range(1, 21))
    frames = []
    for frame in range(1, 289):
        robots = {robot_id: [0.0] * 7 for robot_id in robot_ids}
        if frame == 144:
            robots.pop("R20")
        frames.append({"frame_index": frame, "robots": robots, "workpieces": {}})
    bundle = SimpleNamespace(robot_ids=robot_ids, frames=tuple(frames))

    with pytest.raises(ValueError, match="robot"):
        animation_manifest_from_bundle(bundle)
