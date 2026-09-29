#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-eval}"
case "$MODE" in
  eval|all) MODE="eval" ;;
  --self-test|self-test) MODE="self-test" ;;
  build-memory) MODE="build-memory" ;;
  *)
    echo "Kullanim: bash scripts/train_v07_local_gpu.sh [eval|build-memory|--self-test]" >&2
    echo "Not: strict zero-train FLM model egitmez; bu dosya eski adi koruyan uyumluluk wrapper'idir." >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "self-test" ]]; then
  cat <<EOF
architecture=tokenizer + SemanticCompiler + SemanticMemory + ReasoningCore + ResponseComposer
neural_model_training=false
training_steps=0
checkpoint_required=false
gpu_required=false
tokenizer_training=Kaggle-only
EOF
  exit 0
fi

export PYTHONPATH="$ROOT"

if [[ "$MODE" == "build-memory" ]]; then
  INPUT="${FLM_SEMANTIC_INPUT:-}"
  OUT="${FLM_SEMANTIC_MEMORY:-$ROOT/local-runs/flm-v0.7-zero-train/semantic_memory.json}"
  if [[ -z "$INPUT" ]]; then
    echo "HATA: build-memory icin FLM_SEMANTIC_INPUT ayarla." >&2
    exit 2
  fi
  mkdir -p "$(dirname "$OUT")"
  exec python scripts/build_semantic_memory.py --input "$INPUT" --out "$OUT"
fi

exec python scripts/eval_semantic_core.py
