#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
MODEL_REVISION="${MODEL_REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"

[[ -x "$VIRTUAL_ENV/bin/hf" ]] || { echo "Run scripts/setup_uv.sh first" >&2; exit 1; }
mkdir -p "$HF_MODEL"
"$VIRTUAL_ENV/bin/hf" download "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --local-dir "$HF_MODEL"

test -s "$HF_MODEL/config.json"
test -s "$HF_MODEL/model.safetensors.index.json" || test -s "$HF_MODEL/model.safetensors"
echo "MODEL_READY=$HF_MODEL"
