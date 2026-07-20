#!/bin/bash
#SBATCH --partition=batch_block1
#SBATCH --account=nvr_lacr_llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao/scratch_impl/logs/%x_%j.out

# paper_006_sao — OFFLINE torch_dist → HF conversion (F001: never trust
# in-training --save-hf; probe a step-0 export ≈ base before any trained eval).
# Env: ITER_DIR (…/ckpt/iter_XXXXXXX), OUT_DIR.
set -euxo pipefail

CONTAINER=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/slime_nightly-dev-20260707a.sqsh
SB=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao
SLIME_DIR=$SB/scratch_impl/code/slime
export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface
HF_MODEL=$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

: "${ITER_DIR:?set ITER_DIR}"
: "${OUT_DIR:?set OUT_DIR}"

srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home \
     --no-container-entrypoint \
     bash -c "set -euxo pipefail
        export PYTHONPATH=/root/Megatron-LM:$SLIME_DIR
        export HF_HOME=$HF_HOME
        cd $SLIME_DIR
        source scripts/models/qwen3.5-4B.sh
        python tools/convert_torch_dist_to_hf.py \
           --input-dir $ITER_DIR \
           --output-dir $OUT_DIR \
           --origin-hf-dir $HF_MODEL ${CONVERT_EXTRA:-}"
ls -la "$OUT_DIR"
