#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <blender-bin> <scene.blend> <output-dir>" >&2
  exit 64
fi

BLENDER="$1"
SCENE="$2"
OUT="$3"
FRAMES="$OUT/frames"

[[ -x "$BLENDER" ]] || { echo "Blender binary is not executable: $BLENDER" >&2; exit 2; }
[[ -s "$SCENE" ]] || { echo "Scene file is missing or empty: $SCENE" >&2; exit 2; }

rm -rf "$FRAMES"
mkdir -p "$FRAMES"

"$BLENDER" \
  --background "$SCENE" \
  --disable-autoexec \
  --python factory_robot_3d/blender/render_scene.py \
  -- \
  --frames-dir "$FRAMES"

COUNT="$(find "$FRAMES" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d ' ')"
if [[ "$COUNT" != "288" ]]; then
  echo "Expected 288 rendered frames, got $COUNT" >&2
  exit 3
fi

for frame in 0001 0120 0288; do
  [[ -s "$FRAMES/frame_${frame}.png" ]] || {
    echo "Required rendered frame missing: frame_${frame}.png" >&2
    exit 3
  }
done

cp "$FRAMES/frame_0001.png" "$OUT/preview_wide.png"
cp "$FRAMES/frame_0120.png" "$OUT/preview_close.png"
cp "$FRAMES/frame_0288.png" "$OUT/final_frame.png"

echo "FACTORY_FRAMES_OK count=$COUNT dir=$FRAMES"
