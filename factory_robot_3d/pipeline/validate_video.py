from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


@dataclass(frozen=True)
class VideoValidationReport:
    ok: bool
    errors: tuple[str, ...]


def _rate(value: object) -> float:
    try:
        return float(Fraction(str(value)))
    except Exception:
        return 0.0


def _int_or_zero(value: object) -> int:
    try:
        return int(str(value))
    except Exception:
        return 0


def validate_probe_payload(payload: Mapping[str, Any]) -> VideoValidationReport:
    errors: list[str] = []
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return VideoValidationReport(False, ("ffprobe returned no video stream",))

    stream = streams[0]
    if not isinstance(stream, dict):
        return VideoValidationReport(False, ("ffprobe video stream is invalid",))

    codec = str(stream.get("codec_name", "")).lower()
    if codec != "h264":
        errors.append(f"codec must be h264, got {codec or 'missing'}")

    width = _int_or_zero(stream.get("width"))
    height = _int_or_zero(stream.get("height"))
    if width != 1920:
        errors.append(f"video width must be 1920, got {width}")
    if height != 1080:
        errors.append(f"video height must be 1080, got {height}")

    fps = _rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    if abs(fps - 24.0) > 1e-6:
        errors.append(f"video frame rate must be 24 fps, got {fps:g}")

    frame_count = _int_or_zero(
        stream.get("nb_frames") or stream.get("nb_read_frames")
    )
    if frame_count != 288:
        errors.append(f"video frame count must be 288, got {frame_count}")

    pix_fmt = str(stream.get("pix_fmt", "")).lower()
    if pix_fmt != "yuv420p":
        errors.append(f"pixel format must be yuv420p, got {pix_fmt or 'missing'}")

    format_info = payload.get("format")
    duration_value = None
    if isinstance(format_info, dict):
        duration_value = format_info.get("duration")
    if duration_value is None:
        duration_value = stream.get("duration")
    try:
        duration = float(duration_value)
    except Exception:
        duration = 0.0
    if not (11.9 <= duration <= 12.1):
        errors.append(f"video duration must be about 12.0 seconds, got {duration:.3f}")

    return VideoValidationReport(ok=not errors, errors=tuple(errors))


def probe_video(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"video is missing or empty: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,"
            "nb_frames,nb_read_frames,duration,pix_fmt:"
            "format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for production video validation") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe failed for {path}: {exc.stderr.strip()}"
        ) from exc

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("ffprobe returned an invalid payload")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    args = parser.parse_args(argv)

    payload = probe_video(args.video)
    report = validate_probe_payload(payload)
    print(
        json.dumps(
            {"ok": report.ok, "errors": list(report.errors)},
            sort_keys=True,
        )
    )
    if not report.ok:
        raise SystemExit("video validation failed: " + "; ".join(report.errors))
    print("FACTORY_VIDEO_OK", args.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
