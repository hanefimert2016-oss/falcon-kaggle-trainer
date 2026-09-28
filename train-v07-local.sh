#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# == 0 )); then
  set -- all
fi
exec bash "$ROOT/scripts/train_v07_local_gpu.sh" "$@"
