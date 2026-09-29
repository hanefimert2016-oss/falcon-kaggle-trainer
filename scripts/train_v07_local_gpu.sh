#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-interface}"
case "$MODE" in
  interface|all) MODE="interface" ;;
  --self-test) MODE="self-test" ;;
  *)
    echo "Kullanim: bash scripts/train_v07_local_gpu.sh [interface|all|--self-test]" >&2
    echo "Semantic FLM v0.7'de Main/Coder/ComputerUse icin ayri Transformer egitimi yok." >&2
    exit 2
    ;;
esac

DATA_ROOT="${FLM_LOCAL_DATA_ROOT:-$ROOT/.local-data/v07-semantic-r7}"
TEXT_DIR="$DATA_ROOT/text"
OUT_ROOT="${FLM_LOCAL_OUTPUT_ROOT:-$ROOT/local-runs/flm-v0.7-semantic}"
LOG_DIR="$OUT_ROOT/logs"
VENV="${FLM_LOCAL_VENV:-$ROOT/.venv-flm-v07-interface}"
TEXT_REF="${FLM_V07_TEXT_DATASET:-mertsigma/flm-v07-semantic-text-r7}"

mkdir -p "$DATA_ROOT" "$OUT_ROOT" "$LOG_DIR"

gpu_profile() {
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "HATA: nvidia-smi bulunamadi." >&2
    return 1
  }
  mapfile -t VRAMS < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
  mapfile -t GPUS < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  (("${#VRAMS[@]}" > 0)) || { echo "HATA: NVIDIA GPU bulunamadi." >&2; return 1; }
  MIN_VRAM="${VRAMS[0]}"
  for v in "${VRAMS[@]}"; do (( v < MIN_VRAM )) && MIN_VRAM="$v"; done

  if (( MIN_VRAM >= 23500 )); then
    BATCH=2; ACCUM=8
    GENERAL_SEQ=1024; CODE_SEQ=2048; SFT_SEQ=4096
    PROFILE="24GB+ full curriculum"
  elif (( MIN_VRAM >= 14000 )); then
    BATCH=1; ACCUM=16
    GENERAL_SEQ=1024; CODE_SEQ=1536; SFT_SEQ=3072
    PROFILE="16GB reduced-window"
  else
    BATCH=1; ACCUM=32
    GENERAL_SEQ=768; CODE_SEQ=1024; SFT_SEQ=2048
    PROFILE="8-12GB safe"
  fi

  echo "Semantic FLM tek-Transformer GPU profili: $PROFILE"
  for i in "${!GPUS[@]}"; do
    echo "  GPU $i: ${GPUS[$i]} / ${VRAMS[$i]} MiB"
  done
  echo "  batch=$BATCH accum=$ACCUM general_seq=$GENERAL_SEQ code_seq=$CODE_SEQ sft_seq=$SFT_SEQ"
}

gpu_profile
if [[ "$MODE" == "self-test" ]]; then
  echo "architecture=1x InterfaceTransformer + training-free FLM Core"
  echo "dataset=$TEXT_REF"
  exit 0
fi

if [[ ! -d "$VENV" ]]; then python3 -m venv "$VENV"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools
python -m pip install -q "numpy>=1.26" pillow "tokenizers>=0.20" "kaggle>=2.2.1"

if ! python - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
then
  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
fi

if [[ "${FLM_LOCAL_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  READY=0
  if [[ -s "$TEXT_DIR/sources.json" ]]; then
    if python - "$TEXT_DIR/sources.json" <<'PY'
import json,sys
from pathlib import Path
x=json.load(open(sys.argv[1],encoding="utf-8"))
raise SystemExit(0 if x.get("pipeline_version")=="v0.7-dev-text" and int(x.get("data_revision",0))>=7 else 1)
PY
    then READY=1; fi
  fi
  if (( ! READY )); then
    mkdir -p "$TEXT_DIR"
    rm -rf "$TEXT_DIR"/*
    kaggle datasets download -d "$TEXT_REF" -p "$TEXT_DIR" --unzip
  fi
fi

python - "$TEXT_DIR/sources.json" <<'PY'
import json,sys
from pathlib import Path
x=json.load(open(sys.argv[1],encoding="utf-8"))
s=x.get("stats") or {}
assert x.get("pipeline_version")=="v0.7-dev-text"
assert int(x.get("data_revision",0))>=7
assert int((s.get("main") or {}).get("tokens",0))>=1_000_000_000
assert int((s.get("coder") or {}).get("tokens",0))>=175_000_000
assert int((s.get("semantic_interface_sft") or {}).get("rows",0))>=1_200_000
assert int((s.get("semantic_sft") or {}).get("supervised_tokens",0))>=150_000_000
for name in ("semantic_sft_tokens.u16","semantic_sft_mask.u8"):
    assert (Path(sys.argv[1]).parent/name).is_file(), name
assert int((s.get("tokenizer") or {}).get("vocab_size",0))>=30_000
print("SEMANTIC_FLM_R7_LOCAL_DATA_OK")
PY

export PYTHONPATH="$ROOT"
export FLM_ACCELERATOR=gpu
export FLM_V07_DATA_ROOT="$TEXT_DIR"
export FLM_V07_TEXT_VERSION=v0.7-dev-text
export FLM_INTERFACE_OUTPUT_ROOT="$OUT_ROOT"
export FLM_V07_RESUME=1

export FLM_INTERFACE_SEQ="${FLM_INTERFACE_SEQ:-4096}"
export FLM_INTERFACE_LAYERS="${FLM_INTERFACE_LAYERS:-16}"
export FLM_INTERFACE_HEADS="${FLM_INTERFACE_HEADS:-12}"
export FLM_INTERFACE_KV_HEADS="${FLM_INTERFACE_KV_HEADS:-4}"
export FLM_INTERFACE_EMBD="${FLM_INTERFACE_EMBD:-768}"
export FLM_INTERFACE_BATCH="${FLM_INTERFACE_BATCH:-$BATCH}"
export FLM_INTERFACE_ACCUM="${FLM_INTERFACE_ACCUM:-$ACCUM}"
export FLM_INTERFACE_GENERAL_SEQ="${FLM_INTERFACE_GENERAL_SEQ:-$GENERAL_SEQ}"
export FLM_INTERFACE_CODE_SEQ="${FLM_INTERFACE_CODE_SEQ:-$CODE_SEQ}"
export FLM_INTERFACE_SFT_SEQ="${FLM_INTERFACE_SFT_SEQ:-$SFT_SEQ}"
export FLM_INTERFACE_GENERAL_EPOCHS="${FLM_INTERFACE_GENERAL_EPOCHS:-1.0}"
export FLM_INTERFACE_CODE_EPOCHS="${FLM_INTERFACE_CODE_EPOCHS:-1.0}"
export FLM_INTERFACE_SFT_EPOCHS="${FLM_INTERFACE_SFT_EPOCHS:-1.0}"
export FLM_INTERFACE_GRADIENT_CHECKPOINTING="${FLM_INTERFACE_GRADIENT_CHECKPOINTING:-1}"
export FLM_INTERFACE_CKPT_INTERVAL="${FLM_INTERFACE_CKPT_INTERVAL:-250}"
export FLM_INTERFACE_EVAL_INTERVAL="${FLM_INTERFACE_EVAL_INTERVAL:-500}"
export FLM_INTERFACE_SFT_EVAL_INTERVAL="${FLM_INTERFACE_SFT_EVAL_INTERVAL:-250}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/interface-$STAMP.log"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/latest.log"

cat <<EOF
FLM v0.7 SEMANTIC LOCAL
  trainable_transformers=1
  trainable_model=InterfaceTransformer
  training_free_core=SemanticMemory+ReasoningVM+Planner+Solver+Verifier+CodeEngine+UIPlanner
  data=$TEXT_DIR
  output=$OUT_ROOT
  log=$LOG
EOF

python -u -m flm.train_interface_v07 2>&1 | tee "$LOG"
