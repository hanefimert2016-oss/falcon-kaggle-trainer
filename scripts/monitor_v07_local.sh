#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMORY="${FLM_SEMANTIC_MEMORY:-$ROOT/local-runs/flm-v0.7-zero-train/semantic_memory.json}"

echo "=== FLM v0.7 strict zero-train ==="
echo "neural_model_training=false"
echo "checkpoint_required=false"
echo "gpu_required=false"
echo
echo "=== Semantic memory ==="
if [[ -f "$MEMORY" ]]; then
  ls -lh "$MEMORY"
else
  echo "Henuz semantic memory dosyasi yok: $MEMORY"
fi
echo
echo "=== Core smoke/eval ==="
PYTHONPATH="$ROOT" python "$ROOT/scripts/eval_semantic_core.py"
