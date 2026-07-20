#!/bin/bash
# paper_006_sao — 2-NODE async training launcher (INT-004). Runs once PER NODE
# (srun --nodes=2 --ntasks-per-node=1, each task in its own enroot container).
# SLURM_NODEID 0 = ray HEAD + actor/critic (8 GPU) + submits the training job;
# SLURM_NODEID 1 = ray WORKER + rollout sglang engines (8 GPU).
# Usage: launch_train_2node.sh <ARM_CONFIG.sh>
# NOT set -e: transient ray/curl nonzero returns are normal while the cluster forms.
set -uxo pipefail

ARM_CONFIG=${1:?usage: launch_train_2node.sh <arm_config.sh>}

# Head node IP is computed in the sbatch (on the batch host, which HAS scontrol/
# hostname — the enroot container does NOT: `scontrol: command not found`) and
# passed in via HEAD_IP. enroot uses host networking → host IPs are routable.
HEAD_IP=${HEAD_IP:?HEAD_IP must be exported by submit_train_2node.sbatch}
RAY_PORT=${RAY_PORT:-6379}
# Completion sentinel (unique per job) so the worker exits promptly when the head's
# training job returns, instead of idling to walltime (wastes GPU-h on short jobs).
SENTINEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/sao_scratch/.ray_done_${SLURM_JOB_ID:-0}
DASH_PORT=${DASH_PORT:-8265}

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2

set +x; set -a
source /lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/Research-skills/.env 2>/dev/null || true
set +a; set -x
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

if [[ "${SLURM_NODEID:-0}" != "0" ]]; then
   # ---- WORKER (rollout node) ----
   sleep 25   # let the head bring up the GCS first
   for i in $(seq 60); do
      ray start --address="$HEAD_IP:$RAY_PORT" --num-gpus 8 --disable-usage-stats && break
      sleep 5
   done
   # Stay alive while the head trains; exit as soon as the head signals completion
   # (or if the head process/GCS dies). srun then tears the step down cleanly.
   while [ ! -f "$SENTINEL" ]; do
      ray status --address="$HEAD_IP:$RAY_PORT" >/dev/null 2>&1 || { echo "worker: head GCS gone, exiting"; break; }
      sleep 15
   done
   echo "worker: done (sentinel=$([ -f "$SENTINEL" ] && echo yes || echo no))"
   ray stop --force 2>/dev/null || true
   exit 0
fi

# head removes any stale sentinel (unique-per-job name makes this belt-and-suspenders)
rm -f "$SENTINEL"

# ---- HEAD (actor/critic node) ----
ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
   --dashboard-host=0.0.0.0 --dashboard-port="$DASH_PORT" \
   --num-gpus 8 --disable-usage-stats

# Wait until BOTH nodes' GPUs (16) have registered before submitting.
for i in $(seq 60); do
   NGPU=$(python3 - <<PY 2>/dev/null || echo 0
import ray
ray.init(address="$HEAD_IP:$RAY_PORT")
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
)
   echo "ray cluster GPUs: $NGPU / 16"
   [ "${NGPU:-0}" -ge 16 ] && break
   sleep 5
done

source "$ARM_CONFIG"   # ACTOR_GPUS/ROLLOUT_GPUS (env-overridden to 8) + arg arrays

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

# actor+critic on node 0 (8 GPU), rollout on node 1 (8 GPU) — 16 total in the cluster.
ray job submit --address="http://$HEAD_IP:$DASH_PORT" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 "$SLIME_DIR/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS:-8}" \
   --rollout-num-gpus "${ROLLOUT_GPUS:-8}" \
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
RC=$?
touch "$SENTINEL"   # signal the worker to exit
ray stop --force 2>/dev/null || true
exit $RC
