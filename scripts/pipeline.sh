#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-help}"
shift || true
case "$STAGE" in
  setup) exec "$ROOT/scripts/setup_uv.sh" "$@" ;;
  preflight) exec "$ROOT/scripts/preflight.sh" "$@" ;;
  download) exec "$ROOT/scripts/download_assets.sh" "$@" ;;
  data) exec "$ROOT/scripts/prepare_data.sh" "$@" ;;
  filter) exec "$ROOT/scripts/filter_local.sh" "$@" ;;
  convert) exec "$ROOT/scripts/convert_model.sh" "$@" ;;
  train) exec "$ROOT/scripts/train_local.sh" "$@" ;;
  ready) exec "$ROOT/scripts/training_ready.sh" "$@" ;;
  export) exec "$ROOT/scripts/export_hf.sh" "$@" ;;
  eval) exec "$ROOT/scripts/eval_local.sh" "$@" ;;
  *)
    echo "usage: scripts/pipeline.sh {setup|preflight|download|data|filter|convert|train|ready|export|eval} [...]"
    echo "Stages are intentionally explicit because filtering and full training consume substantial GPU time."
    exit 2
    ;;
esac
