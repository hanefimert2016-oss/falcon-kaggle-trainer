#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
case "$MODE" in
  all|main|coder|computer_use) ;;
  --self-test)
    MODE="self-test"
    ;;
  *)
    echo "Kullanim: bash scripts/train_v07_local_gpu.sh [all|main|coder|computer_use|--self-test]" >&2
    exit 2
    ;;
esac

DATA_ROOT="${FLM_LOCAL_DATA_ROOT:-$ROOT/.local-data/v07}"
TEXT_DIR="$DATA_ROOT/text"
COMPUTER_DIR="$DATA_ROOT/computer"
OUT_ROOT="${FLM_LOCAL_OUTPUT_ROOT:-$ROOT/local-runs/flm-v0.7-r2}"
LOG_DIR="$OUT_ROOT/logs"
VENV="${FLM_LOCAL_VENV:-$ROOT/.venv-flm-v07}"

TEXT_REF="${FLM_V07_TEXT_DATASET:-mertsigma/flm-v07-dev-text}"
COMPUTER_REF="${FLM_V07_COMPUTER_DATASET:-mertsigma/flm-v07-dev-computer}"

mkdir -p "$DATA_ROOT" "$OUT_ROOT" "$LOG_DIR"

FREE_KB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
FREE_GB=$((FREE_KB / 1024 / 1024))
if (( FREE_GB < 45 )); then
  echo "UYARI: v0.7 r2 veri + checkpointler için rahatça 45+ GB boş alan ister; mevcut yaklaşık ${FREE_GB} GB."
fi

gpu_profile() {
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "HATA: nvidia-smi bulunamadi. NVIDIA surucusunu kurup tekrar calistir." >&2
    return 1
  }

  mapfile -t VRAMS < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
  mapfile -t GPUS < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  (("${#VRAMS[@]}" > 0)) || {
    echo "HATA: NVIDIA GPU bulunamadi." >&2
    return 1
  }

  MIN_VRAM="${VRAMS[0]}"
  for v in "${VRAMS[@]}"; do
    (( v < MIN_VRAM )) && MIN_VRAM="$v"
  done
  GPU_COUNT="${#VRAMS[@]}"

  # 4K context profile. Keep enough accumulation for stable global batches.
  if (( MIN_VRAM >= 80000 )); then
    TEXT_BATCH=16
    TEXT_ACCUM=2
    CU_BATCH=16
    PROFILE="80GB+ 4K"
  elif (( MIN_VRAM >= 40000 )); then
    TEXT_BATCH=8
    TEXT_ACCUM=4
    CU_BATCH=8
    PROFILE="40GB+ 4K"
  elif (( MIN_VRAM >= 23500 )); then
    TEXT_BATCH=4
    TEXT_ACCUM=8
    CU_BATCH=4
    PROFILE="24GB 4K"
  elif (( MIN_VRAM >= 15500 )); then
    TEXT_BATCH=2
    TEXT_ACCUM=16
    CU_BATCH=2
    PROFILE="16GB 4K"
  else
    TEXT_BATCH=1
    TEXT_ACCUM=32
    CU_BATCH=1
    PROFILE="8-12GB 4K-safe"
  fi

  if (( MIN_VRAM < 40000 )); then
    TEXT_GRAD_CKPT=1
  else
    TEXT_GRAD_CKPT=0
  fi

  # DataParallel needs at least one sample per participating GPU.
  if (( GPU_COUNT > 1 && TEXT_BATCH < GPU_COUNT )); then
    TEXT_BATCH="$GPU_COUNT"
    TEXT_ACCUM=$(( (16 + TEXT_BATCH - 1) / TEXT_BATCH ))
  fi

  echo "GPU profili: $PROFILE"
  for i in "${!GPUS[@]}"; do
    echo "  GPU $i: ${GPUS[$i]} / ${VRAMS[$i]} MiB"
  done
  echo "  text batch=$TEXT_BATCH accum=$TEXT_ACCUM effective=$((TEXT_BATCH * TEXT_ACCUM))"
  echo "  computer-use batch=$CU_BATCH"
  echo "  activation-checkpointing=$TEXT_GRAD_CKPT"
}

if [[ "$MODE" == "self-test" ]]; then
  gpu_profile
  echo "ROOT=$ROOT"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "OUT_ROOT=$OUT_ROOT"
  exit 0
fi

gpu_profile

if [[ ! -d "$VENV" ]]; then
  echo "Python sanal ortami olusturuluyor: $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools
python -m pip install -q "numpy>=1.26" pillow "tokenizers>=0.20" "kaggle>=2.2.1"

if ! python - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
then
  echo "CUDA PyTorch kuruluyor (cu128 wheel)..."
  python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch CUDA GPU'yu goremiyor"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory/1024**3,2), "GiB")
PY

download_dataset() {
  local ref="$1"
  local dest="$2"
  local version="$3"

  if [[ -s "$dest/sources.json" ]] && python - "$dest/sources.json" "$version" <<'PY'
import json,sys
p,v=sys.argv[1:]
x=json.load(open(p,encoding="utf-8"))
raise SystemExit(0 if x.get("pipeline_version")==v and int(x.get("data_revision",0))>=2 else 1)
PY
  then
    echo "Hazir dataset kullaniliyor: $dest"
    return 0
  fi

  echo "Dataset indiriliyor: $ref"
  rm -rf "$dest"
  mkdir -p "$dest"
  kaggle datasets download -d "$ref" -p "$dest" --unzip
  test -s "$dest/sources.json"
}

if [[ "${FLM_LOCAL_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  NEED_DOWNLOAD=0
  [[ -s "$TEXT_DIR/sources.json" ]] || NEED_DOWNLOAD=1
  [[ -s "$COMPUTER_DIR/sources.json" ]] || NEED_DOWNLOAD=1
  if (( NEED_DOWNLOAD )); then
    if ! kaggle datasets files "$TEXT_REF" --page-size 1 >/dev/null 2>&1; then
      cat >&2 <<'EOF'
HATA: Kaggle dataset kimlik dogrulamasi yok.
Bu Kaggle'da eğitim YAPMAZ; yalnızca hazırlanmış v0.7 verisini PC'ye bir kez indirir.
Kaggle API tokenini ~/.kaggle/kaggle.json ile veya KAGGLE_API_TOKEN ortam değişkeniyle tanımla.
Elinde dataset zaten varsa FLM_LOCAL_DATA_ROOT=... FLM_LOCAL_SKIP_DOWNLOAD=1 kullanabilirsin.
EOF
      exit 3
    fi
  fi
  download_dataset "$TEXT_REF" "$TEXT_DIR" "v0.7-dev-text"
  download_dataset "$COMPUTER_REF" "$COMPUTER_DIR" "v0.7-dev-computer"
fi

python - "$TEXT_DIR/sources.json" "$COMPUTER_DIR/sources.json" <<'PY'
import json,sys
t=json.load(open(sys.argv[1],encoding="utf-8"))
c=json.load(open(sys.argv[2],encoding="utf-8"))
ts=t["stats"]; cs=c["stats"]
assert t["pipeline_version"]=="v0.7-dev-text" and int(t.get("data_revision",0))>=2
assert c["pipeline_version"]=="v0.7-dev-computer" and int(c.get("data_revision",0))>=2
assert ts["main"]["tokens"] >= 1_000_000_000
assert ts["coder"]["tokens"] >= 175_000_000
assert ts["main_sft"]["supervised_tokens"] >= 300_000_000
assert ts["coder_sft"]["supervised_tokens"] >= 75_000_000
assert ts["coder_sft_agentic"]["rows"] >= 2500
assert ts["coder_sft_agentic"]["plan_turns"] >= 5000
assert ts["tokenizer"]["vocab_size"] >= 30_000
assert cs["examples"] >= 40_000
ops=cs["ops"]
for op,n in {"CLICK":20000,"KEY":1000,"TYPE":1000,"SCROLL":750,"DRAG":300,"RIGHT_CLICK":200,"DOUBLE_CLICK":200,"DONE":500}.items():
    assert ops.get(op,0)>=n,(op,ops)
print("V07_LOCAL_DATA_OK")
print("main_tokens",ts["main"]["tokens"])
print("coder_tokens",ts["coder"]["tokens"])
print("main_sft_tokens",ts["main_sft"]["tokens"])
print("coder_sft_tokens",ts["coder_sft"]["tokens"])
print("computer_examples",cs["examples"])
PY

export PYTHONPATH="$ROOT"
export FLM_ACCELERATOR=gpu
export FLM_V07_DATA_ROOT="$DATA_ROOT"
export FLM_V07_TEXT_VERSION=v0.7-dev-text
export FLM_V07_COMPUTER_VERSION=v0.7-dev-computer
export FLM_V07_OUTPUT_ROOT="$OUT_ROOT"
export FLM_V07_RESUME=1

export FLM_V07_MAIN_SEQ="${FLM_V07_MAIN_SEQ:-4096}"
export FLM_V07_MAIN_LAYERS="${FLM_V07_MAIN_LAYERS:-14}"
export FLM_V07_MAIN_HEADS="${FLM_V07_MAIN_HEADS:-12}"
export FLM_V07_MAIN_EMBD="${FLM_V07_MAIN_EMBD:-768}"
export FLM_V07_CODER_SEQ="${FLM_V07_CODER_SEQ:-4096}"
export FLM_V07_CODER_LAYERS="${FLM_V07_CODER_LAYERS:-14}"
export FLM_V07_CODER_HEADS="${FLM_V07_CODER_HEADS:-12}"
export FLM_V07_CODER_EMBD="${FLM_V07_CODER_EMBD:-768}"
export FLM_V07_POSITION_ENCODING="${FLM_V07_POSITION_ENCODING:-rope}"
export FLM_V07_CODER_INIT_FROM_MAIN="${FLM_V07_CODER_INIT_FROM_MAIN:-1}"
export FLM_V07_CU_INIT_FROM_MAIN="${FLM_V07_CU_INIT_FROM_MAIN:-1}"

export FLM_V07_TEXT_BATCH="${FLM_V07_TEXT_BATCH:-$TEXT_BATCH}"
export FLM_V07_TEXT_ACCUM="${FLM_V07_TEXT_ACCUM:-$TEXT_ACCUM}"
export FLM_V07_GRADIENT_CHECKPOINTING="${FLM_V07_GRADIENT_CHECKPOINTING:-$TEXT_GRAD_CKPT}"
export FLM_V07_MAIN_STEPS="${FLM_V07_MAIN_STEPS:-0}"
export FLM_V07_MAIN_SFT_STEPS="${FLM_V07_MAIN_SFT_STEPS:-0}"
export FLM_V07_CODER_STEPS="${FLM_V07_CODER_STEPS:-0}"
export FLM_V07_CODER_SFT_STEPS="${FLM_V07_CODER_SFT_STEPS:-0}"
export FLM_V07_MAIN_PRETRAIN_EPOCHS="${FLM_V07_MAIN_PRETRAIN_EPOCHS:-1.0}"
export FLM_V07_MAIN_SFT_EPOCHS="${FLM_V07_MAIN_SFT_EPOCHS:-1.0}"
export FLM_V07_CODER_PRETRAIN_EPOCHS="${FLM_V07_CODER_PRETRAIN_EPOCHS:-1.0}"
export FLM_V07_CODER_SFT_EPOCHS="${FLM_V07_CODER_SFT_EPOCHS:-1.25}"

export FLM_V07_CU_IMAGE="${FLM_V07_CU_IMAGE:-224}"
export FLM_V07_CU_TASK_LEN="${FLM_V07_CU_TASK_LEN:-512}"
export FLM_V07_CU_PAYLOAD_LEN="${FLM_V07_CU_PAYLOAD_LEN:-128}"
export FLM_V07_CU_EMBD="${FLM_V07_CU_EMBD:-768}"
export FLM_V07_CU_HEADS="${FLM_V07_CU_HEADS:-12}"
export FLM_V07_CU_TEXT_LAYERS="${FLM_V07_CU_TEXT_LAYERS:-4}"
export FLM_V07_CU_VISION_LAYERS="${FLM_V07_CU_VISION_LAYERS:-8}"
export FLM_V07_CU_BATCH="${FLM_V07_CU_BATCH:-$CU_BATCH}"
export FLM_V07_CU_STEPS="${FLM_V07_CU_STEPS:-30000}"

export FLM_V07_EVAL_BATCHES="${FLM_V07_EVAL_BATCHES:-24}"
export FLM_V07_CU_EVAL_EXAMPLES="${FLM_V07_CU_EVAL_EXAMPLES:-512}"
export FLM_V07_TEXT_EVAL_INTERVAL="${FLM_V07_TEXT_EVAL_INTERVAL:-1000}"
export FLM_V07_SFT_EVAL_INTERVAL="${FLM_V07_SFT_EVAL_INTERVAL:-500}"
export FLM_V07_TEXT_CHECKPOINT_INTERVAL="${FLM_V07_TEXT_CHECKPOINT_INTERVAL:-1000}"

# Helps reduce fragmentation on long CUDA jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/train-$MODE-$STAMP.log"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/latest.log"

cat <<EOF
FLM v0.7 LOCAL GPU
  mode=$MODE
  data=$DATA_ROOT
  output=$OUT_ROOT
  log=$LOG
  resume=$FLM_V07_RESUME
  main=$FLM_V07_MAIN_STEPS + sft=$FLM_V07_MAIN_SFT_STEPS
  coder=$FLM_V07_CODER_STEPS + sft=$FLM_V07_CODER_SFT_STEPS
  computer_use=$FLM_V07_CU_STEPS
Monitor:
  watch -n 1 nvidia-smi
  tail -f "$LOG"
EOF

python -u -m flm.train_v07 --only "$MODE" 2>&1 | tee "$LOG"
