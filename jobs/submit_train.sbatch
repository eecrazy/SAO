#!/bin/bash
#SBATCH --partition=batch_block1
#SBATCH --account=nvr_lacr_llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao/scratch_impl/logs/%x_%j.out

# paper_006_sao — one 4h training segment. Chain with jobs/chain.sh (singleton).
# Required env (via --export): ARM_CONFIG. Optional: SMOKE, RUN_TAG, NUM_ROLLOUT,
# SAO_MAX_WEIGHT_STALENESS, ...
set -euxo pipefail

CONTAINER=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/slime_nightly-dev-20260707a.sqsh
SB=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao

srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home \
     --no-container-entrypoint \
     bash "$SB/scratch_impl/jobs/launch_train.sh" "${ARM_CONFIG:?set ARM_CONFIG}"

# prune old dist ckpts (keep newest 3 iters per role dir) to cap disk
RUNS=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/sao_scratch/runs
for d in "$RUNS"/*/ckpt "$RUNS"/*/critic_ckpt; do
   [ -d "$d" ] || continue
   ls -d "$d"/iter_* 2>/dev/null | sort -V | head -n -3 | xargs -r rm -rf
done
