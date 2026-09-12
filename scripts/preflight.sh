#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

[[ -x "$VIRTUAL_ENV/bin/python" ]] || { echo "Missing $VIRTUAL_ENV; run scripts/setup_uv.sh" >&2; exit 1; }
[[ -d "$SLIME_DIR" ]] || { echo "Missing pinned slime source" >&2; exit 1; }
[[ -d "$MEGATRON_DIR/megatron/training" ]] || { echo "Missing pinned Megatron-LM source" >&2; exit 1; }

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ "$GPU_COUNT" -eq 8 ]] || { echo "Expected 8 GPUs, found $GPU_COUNT" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader

"$VIRTUAL_ENV/bin/python" - <<'PY'
import importlib
import pathlib
import sys

import torch

expected = {
    "python": (3, 12),
    "torch": "2.11.0",
    "cuda": "13.0",
    "gpus": 8,
    "capability": (9, 0),
}
assert sys.version_info[:2] == expected["python"], sys.version
assert torch.__version__.startswith(expected["torch"]), torch.__version__
assert torch.version.cuda.startswith(expected["cuda"]), torch.version.cuda
assert torch.cuda.device_count() == expected["gpus"], torch.cuda.device_count()
assert {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())} == {expected["capability"]}

for name in (
    "ray",
    "sglang",
    "sglang_router",
    "transformer_engine",
    "megatron.core",
    "slime",
    "sao_plugin.dis_loss",
    "sao_plugin.adaptive_gae",
    "sao_plugin.reward_math",
    "sao_plugin.async_staleness",
):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'OK')} ({getattr(module, '__file__', '')})")

from sao_plugin.reward_math import grade_response
assert grade_response(r"The answer is \boxed{42}", "42") == 1
assert grade_response(r"The answer is \boxed{41}", "42") == 0
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("reward grader: OK")
PY

rg -q 'SAO_CRITIC_UPDATES_PER_STEP' "$SLIME_DIR/slime/backends/megatron_utils/actor.py"
rg -q 'rollout_max_response_len - sample.response_length' "$SLIME_DIR/slime/rollout/sglang_rollout.py"
rg -q 'allow_partial_load=True' "$MEGATRON_DIR/megatron/core/dist_checkpointing/strategies/torch.py"

for path in "$HF_MODEL/config.json" "$TORCH_DIST/latest_checkpointed_iteration.txt" "$DATA"; do
  if [[ -e "$path" ]]; then
    echo "asset OK: $path"
  else
    echo "asset pending: $path"
  fi
done

echo "SAO_PREFLIGHT_OK"
