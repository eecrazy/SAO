#!/bin/bash
#SBATCH --partition=batch_block1
#SBATCH --account=nvr_lacr_llm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao/scratch_impl/logs/%x_%j.out

# paper_006_sao — one 4h 2-NODE training segment (INT-004: full 24k×1000 on 16 GPU).
# Chain with jobs/chain.sh (singleton). Required env (via --export): ARM_CONFIG.
# Recommended overrides for the 2-node layout: ACTOR_GPUS=8 ROLLOUT_GPUS=8 TP=4
# USE_DIST_OPT=1 MAX_TOKENS_PER_GPU=24576 SGLANG_CONCURRENCY=32 (more rollout GPUs
# → KV headroom to raise concurrency back up).
set -uxo pipefail

CONTAINER=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/slime_nightly-dev-20260707a.sqsh
SB=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao

# The batch script runs on the FIRST allocated node (= the ray head), which HAS
# hostname/scontrol (the enroot container does not). Resolve the head IP here and
# export it so both containers can reach the ray GCS. Prefer the InfiniBand IP
# (ib) if present for fast inter-node NCCL, else the primary IP.
HEAD_IP=$(getent hosts "$(hostname)" 2>/dev/null | awk '{print $1}' | head -1)
[ -z "$HEAD_IP" ] && HEAD_IP=$(hostname -I | awk '{print $1}')
export HEAD_IP
echo "HEAD_IP=$HEAD_IP (batch host $(hostname); nodes $SLURM_JOB_NODELIST)"

# One task per node, each in its own container; launch_train_2node.sh branches on
# SLURM_NODEID (0=head/actor, 1=worker/rollout) and forms one ray cluster.
srun --nodes=2 --ntasks-per-node=1 \
     --export=ALL,HEAD_IP="$HEAD_IP" \
     --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home \
     --no-container-entrypoint \
     bash "$SB/scratch_impl/jobs/launch_train_2node.sh" "${ARM_CONFIG:?set ARM_CONFIG}"

# prune old dist ckpts (keep newest 3 iters per role dir) to cap disk
RUNS=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/sao_scratch/runs
for d in "$RUNS"/*/ckpt "$RUNS"/*/critic_ckpt; do
   [ -d "$d" ] || continue
   ls -d "$d"/iter_* 2>/dev/null | sort -V | head -n -3 | xargs -r rm -rf
done
