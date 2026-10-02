#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <output-dir>" >&2
  exit 64
fi

OUT="$1"
FRAMES="$OUT/frames"
VIDEO="$OUT/factory_robot_cinematic.mp4"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required" >&2; exit 2; }
[[ -s "$FRAMES/frame_0001.png" ]] || { echo "rendered frames are missing" >&2; exit 2; }
[[ -s "$FRAMES/frame_0288.png" ]] || { echo "final rendered frame is missing" >&2; exit 2; }

ffmpeg -y -hide_banner -loglevel warning \
  -framerate 24 \
  -start_number 1 \
  -i "$FRAMES/frame_%04d.png" \
  -frames:v 288 \
  -c:v libx264 \
  -preset slow \
  -crf 17 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$VIDEO"

[[ -s "$VIDEO" ]] || { echo "encoded video is missing or empty" >&2; exit 3; }
echo "FACTORY_VIDEO_ENCODE_OK $VIDEO"
