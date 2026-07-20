#!/bin/bash
# paper_006_sao — in-container training launcher (runs under srun inside
# slime_nightly-dev-20260707a.sqsh). Usage: launch_train.sh <ARM_CONFIG.sh>
# Env: SMOKE=1 → tiny 3-rollout shakedown into a throwaway ckpt dir.
set -euxo pipefail

ARM_CONFIG=${1:?usage: launch_train.sh <arm_config.sh>}

# clean any leftover ray/sglang from a previous segment on this node
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

# secrets (WANDB_API_KEY, HF_TOKEN) — never hardcode
set +x
set -a
source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/Research-skills/.env 2>/dev/null || true
set +a
set -x

export PYTHONUNBUFFERED=1

# smoke mode: 3 rollouts, short responses, throwaway ckpts, no critic warmup skip.
# Default DATA to candidates.jsonl (always present) rather than common.sh's
# pool.jsonl default, which only exists AFTER the filter campaign emits it — a
# bare `SMOKE=1` resubmit must not depend on the final pool.
if [[ "${SMOKE:-0}" == "1" ]]; then
   export RUN_TAG=${RUN_TAG:-smoke}
   export NUM_ROLLOUT=${NUM_ROLLOUT:-3}
   export MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024}
   export SAVE_INTERVAL=${SAVE_INTERVAL:-2}
   export NUM_CRITIC_ONLY_STEPS=${NUM_CRITIC_ONLY_STEPS:-1}
   export DATA=${DATA:-/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/sao_scratch/data/candidates.jsonl}
fi

source "$ARM_CONFIG"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

NUM_GPUS=${NUM_GPUS:-8}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" --disable-usage-stats

# PYTHONPATH: image Megatron + pinned slime clone (shadows any pip slime) +
# $CODE so `sao_plugin` imports inside ray workers.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${SLIME_DIR}:${CODE}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"HF_HOME\": \"${HF_HOME}\",
    \"SAO_CRITIC_UPDATES_PER_STEP\": \"${SAO_CRITIC_UPDATES_PER_STEP:-1}\",
    \"SAO_GAE_LAMBDA_ALPHA\": \"${SAO_GAE_LAMBDA_ALPHA:-1.5}\",
    \"SAO_MAX_WEIGHT_STALENESS\": \"${SAO_MAX_WEIGHT_STALENESS:-0}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 "$SLIME_DIR/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$ACTOR_GPUS" \
   --rollout-num-gpus "$ROLLOUT_GPUS" \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_COMMON[@]}" \
   "${DIS_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}
