#!/usr/bin/env bash
set -euo pipefail

CKPT=${CKPT:?set CKPT}
ASSETS=${ASSETS:?set ASSETS}
IMAGE=${IMAGE:-examples/03/image.jpg}
ACTION=${ACTION:-examples/03}
AREA=${AREA:-114688}
FRAMES=${FRAMES:-81}
LOCAL_ATTN=${LOCAL_ATTN:-6}
SINK=${SINK:-2}
CHUNK=${CHUNK:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

python generate.py \
  --task i2v-1.3B \
  --infer_mode causal_fast \
  --size 480*832 \
  --max_area_override "$AREA" \
  --ckpt_dir "$CKPT" \
  --assets_dir "$ASSETS" \
  --image "$IMAGE" \
  --action_path "$ACTION" \
  --frame_num "$FRAMES" \
  --chunk_size "$CHUNK" \
  --local_attn_size "$LOCAL_ATTN" \
  --sink_size "$SINK" \
  --t5_cpu \
  --offload_model False \
  --prompt "Interactive world, stable geometry, coherent motion, realistic detail"
