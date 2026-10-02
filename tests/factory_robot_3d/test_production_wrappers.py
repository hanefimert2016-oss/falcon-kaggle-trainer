from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_render_and_encode_wrappers_exist_with_production_contract():
    root = Path("factory_robot_3d/pipeline")
    render = (root / "render_frames.sh").read_text(encoding="utf-8")
    encode = (root / "encode_video.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in render
    assert "frame_####" in render
    assert "preview_wide.png" in render
    assert "preview_close.png" in render
    assert "final_frame.png" in render
    assert "factory_robot_3d/blender/render_scene.py" in render
    assert "--python" in render

    assert "set -euo pipefail" in encode
    assert "ffmpeg" in encode
    assert "frame_%04d.png" in encode
    assert "libx264" in encode
    assert "yuv420p" in encode
    assert "factory_robot_cinematic.mp4" in encode
    assert "-framerate 24" in encode


def test_video_probe_accepts_exact_production_video_contract():
    from factory_robot_3d.pipeline.validate_video import validate_probe_payload

    payload = {
        "streams": [{
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "24/1",
            "avg_frame_rate": "24/1",
            "nb_frames": "288",
            "duration": "12.000000",
            "pix_fmt": "yuv420p",
        }],
        "format": {"duration": "12.000000"},
    }

    report = validate_probe_payload(payload)

    assert report.ok is True
    assert report.errors == ()


@pytest.mark.parametrize(
    "patch, expected",
    [
        ({"width": 1280}, "1920"),
        ({"height": 720}, "1080"),
        ({"r_frame_rate": "30/1", "avg_frame_rate": "30/1"}, "24"),
        ({"nb_frames": "287"}, "288"),
        ({"codec_name": "hevc"}, "h264"),
        ({"pix_fmt": "yuv444p"}, "yuv420p"),
    ],
)
def test_video_probe_rejects_nonproduction_video_contract(patch, expected):
    from factory_robot_3d.pipeline.validate_video import validate_probe_payload

    stream = {
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "r_frame_rate": "24/1",
        "avg_frame_rate": "24/1",
        "nb_frames": "288",
        "duration": "12.000000",
        "pix_fmt": "yuv420p",
    }
    stream.update(patch)
    report = validate_probe_payload({
        "streams": [stream],
        "format": {"duration": stream.get("duration", "12.000000")},
    })

    assert report.ok is False
    assert any(expected in error for error in report.errors)


def test_production_pipeline_runs_gpu_probe_before_scene_build(monkeypatch, tmp_path):
    import factory_robot_3d.pipeline.run_production as module

    commands: list[list[str]] = []

    def fake_run(command, *, log=None, env=None):
        command = list(command)
        commands.append(command)
        joined = " ".join(command)
        if "factory_robot_3d.run_simulation" in joined:
            (tmp_path / "summary.json").write_text(json.dumps({"success": True}))
        if "encode_video.sh" in joined:
            for name in (
                "factory_robot_cinematic.mp4",
                "factory_config.json",
                "animation.json",
                "telemetry.csv",
                "summary.json",
                "final_frame.png",
                "preview_wide.png",
                "preview_close.png",
                "render.log",
            ):
                path = tmp_path / name
                if not path.exists() or path.stat().st_size == 0:
                    path.write_bytes(b"x")

    class Report:
        ok = True
        errors = ()
        files = ()

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "validate_artifacts", lambda _root: Report())

    module.run_production(
        tmp_path,
        blender_bin=Path("/opt/blender/blender"),
        python_executable="python",
    )

    probe_index = next(
        i for i, cmd in enumerate(commands)
        if "factory_robot_3d/blender/probe_gpu.py" in cmd
    )
    build_index = next(
        i for i, cmd in enumerate(commands)
        if "factory_robot_3d/blender/build_scene.py" in cmd
    )
    assert probe_index < build_index


def test_render_driver_reconfigures_gpu_and_renders_animation():
    driver = Path("factory_robot_3d/blender/render_scene.py").read_text(encoding="utf-8")

    assert "configure_cycles" in driver
    assert "bpy.ops.render.render(animation=True)" in driver
    assert "FACTORY_RENDER_RUNTIME_DEVICE" in driver
    assert "frame_" in driver


def test_production_surfaces_render_log_tail_on_blender_failure():
    source = Path("factory_robot_3d/pipeline/run_production.py").read_text(encoding="utf-8")

    assert "RENDER_LOG_TAIL_BEGIN" in source
    assert "RENDER_LOG_TAIL_END" in source
    assert "CalledProcessError" in source
