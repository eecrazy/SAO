#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PYTHON="$VIRTUAL_ENV/bin/python"
ITER_DIR="${1:?usage: export_hf.sh RUN_ITER_DIR OUTPUT_HF_DIR}"
OUT_DIR="${2:?usage: export_hf.sh RUN_ITER_DIR OUTPUT_HF_DIR}"

test -d "$ITER_DIR"
mkdir -p "$(dirname -- "$OUT_DIR")"
ITER_DIR="$(cd -- "$ITER_DIR" && pwd)"
OUT_DIR="$(cd -- "$(dirname -- "$OUT_DIR")" && pwd)/$(basename -- "$OUT_DIR")"
cd "$SLIME_DIR"
source "$SLIME_DIR/scripts/models/qwen3.5-4B.sh"
"$PYTHON" "$SLIME_DIR/tools/convert_torch_dist_to_hf.py" \
  --input-dir "$ITER_DIR" \
  --output-dir "$OUT_DIR" \
  --origin-hf-dir "$HF_MODEL" \
  --add-missing-from-origin-hf \
  ${CONVERT_EXTRA:-}
test -s "$OUT_DIR/config.json"
test -s "$OUT_DIR/model.safetensors.index.json" || test -s "$OUT_DIR/model.safetensors"
echo "HF_CHECKPOINT_READY=$OUT_DIR"
