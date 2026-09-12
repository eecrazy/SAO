#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PYTHON="$VIRTUAL_ENV/bin/python"

test -s "$HF_MODEL/config.json"
mkdir -p "$TORCH_DIST"
cd "$SLIME_DIR"
source "$SLIME_DIR/scripts/models/qwen3.5-4B.sh"
"$PYTHON" "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_MODEL" \
  --save "$TORCH_DIST"

test -s "$TORCH_DIST/latest_checkpointed_iteration.txt"
echo "MEGATRON_CHECKPOINT_READY=$TORCH_DIST"
