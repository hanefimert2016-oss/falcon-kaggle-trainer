#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${FLM_LOCAL_OUTPUT_ROOT:-$ROOT/local-runs/flm-v0.7-r2}"
LOG="$OUT_ROOT/logs/latest.log"

echo "=== GPU ==="
nvidia-smi || true
echo
echo "=== Latest checkpoints ==="
find "$OUT_ROOT" -maxdepth 2 -type f \( -name '*.pt' -o -name '*.json' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %10s %p\n' 2>/dev/null | sort | tail -30 || true
echo
echo "=== Latest training log ==="
if [[ -e "$LOG" ]]; then
  tail -80 "$LOG"
else
  echo "Henuz log yok: $LOG"
fi
