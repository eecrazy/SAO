#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
ARM="${1:?usage: train_local.sh sao|grpo_dis [smoke|full]}"
MODE="${2:-full}"
case "$ARM" in
  sao|grpo_dis) ;;
  *) echo "arm must be sao or grpo_dis" >&2; exit 2 ;;
esac
case "$MODE" in
  smoke|full) ;;
  *) echo "mode must be smoke or full" >&2; exit 2 ;;
esac

CONFIG="$ROOT/sao_plugin/configs/$ARM.sh"
PYTHON="$VIRTUAL_ENV/bin/python"
RAY="$VIRTUAL_ENV/bin/ray"

if [[ "$MODE" == smoke ]]; then
  export RUN_TAG="${RUN_TAG:-smoke}"
  export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
  export MAX_RESPONSE_LEN="${SMOKE_MAX_RESPONSE_LEN:-1024}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
  export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-1}"
  export DATA="${SMOKE_DATA:-$ROOT/data/candidates.jsonl}"
fi
if [[ -z "${MAX_TOKENS_PER_GPU+x}" ]]; then
  if [[ "$ARM" == sao ]]; then
    export MAX_TOKENS_PER_GPU="${SAO_MAX_TOKENS_PER_GPU:-12288}"
  else
    export MAX_TOKENS_PER_GPU="${GRPO_MAX_TOKENS_PER_GPU:-32768}"
  fi
fi
test -s "$DATA"
test -s "$HF_MODEL/config.json"
test -s "$TORCH_DIST/latest_checkpointed_iteration.txt"

source "$CONFIG"
[[ $((ACTOR_GPUS + ROLLOUT_GPUS)) -eq "$NUM_GPUS" ]]

TRAIN_LOG="${TRAIN_LOG:-$ROOT/logs/train_${ARM}_${MODE}.log}"
mkdir -p "$(dirname -- "$TRAIN_LOG")"
exec > >(tee -a "$TRAIN_LOG") 2>&1
echo "[$(date --iso-8601=seconds)] starting arm=$ARM mode=$MODE log=$TRAIN_LOG"
echo "run_dir=$RUN_DIR data=$DATA num_rollout=$NUM_ROLLOUT actor_gpus=$ACTOR_GPUS rollout_gpus=$ROLLOUT_GPUS"

cleanup() { "$RAY" stop --force >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
cleanup

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | rg -o 'NV[0-9]+' | wc -l)"
if [[ "$NVLINK_COUNT" -gt 0 ]]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
"$RAY" start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" \
  --dashboard-host=127.0.0.1 --dashboard-port=8265 --disable-usage-stats

RUNTIME_ENV_JSON="$("$PYTHON" - <<PY
import json
print(json.dumps({"env_vars": {
  "PATH": "$VIRTUAL_ENV/bin:" + __import__("os").environ["PATH"],
  "PYTHONPATH": "$PYTHONPATH",
  "CUDA_HOME": "$CUDA_HOME",
  "TORCH_CUDA_ARCH_LIST": "$TORCH_CUDA_ARCH_LIST",
  "CUDA_DEVICE_MAX_CONNECTIONS": "1",
  "NCCL_NVLS_ENABLE": "$HAS_NVLINK",
  "HF_HOME": "$HF_HOME",
  "SAO_CRITIC_UPDATES_PER_STEP": "${SAO_CRITIC_UPDATES_PER_STEP:-1}",
  "SAO_GAE_LAMBDA_ALPHA": "${SAO_GAE_LAMBDA_ALPHA:-1.5}",
  "SAO_MAX_WEIGHT_STALENESS": "$SAO_MAX_WEIGHT_STALENESS",
}}))
PY
)"

"$RAY" job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json "$RUNTIME_ENV_JSON" -- \
  "$PYTHON" "$SLIME_DIR/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$ACTOR_GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_COMMON[@]}" \
  "${DIS_ARGS[@]}" \
  "${ALGO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
