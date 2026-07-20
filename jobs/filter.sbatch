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

# paper_006_sao — difficulty-filter campaign segment (resumable; chain with
# --dependency=singleton --job-name=sao-filter until results cover candidates).
# Serves Qwen3.5-4B with sglang (dp=8 × tp=1) and runs filter_campaign.py
# (n=8, temp 1.0, 24k) against it. Env overrides: CANDIDATES, RESULTS, FILTER_N.
set -euxo pipefail

CONTAINER=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/slime_nightly-dev-20260707a.sqsh
SB=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao
CODE=$SB/scratch_impl/code
export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface
HF_MODEL=$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
RUNS=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/sao_scratch
CANDIDATES=${CANDIDATES:-$RUNS/data/candidates.jsonl}
RESULTS=${RESULTS:-$RUNS/data/filter_results.jsonl}
mkdir -p "$RUNS/data"

srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home \
     --no-container-entrypoint \
     bash -c "set -euxo pipefail
        export HF_HOME=$HF_HOME
        export PYTHONPATH=$CODE
        pkill -9 sglang 2>/dev/null || true; sleep 2
        python -m sglang.launch_server \
           --model-path $HF_MODEL --served-model-name qwen3.5-4b \
           --dp-size 8 --tp-size 1 --mem-fraction-static 0.85 \
           --host 127.0.0.1 --port 30000 --log-level warning &
        SERVER_PID=\$!
        for i in \$(seq 120); do
           curl -sf http://127.0.0.1:30000/health >/dev/null && break
           sleep 5
        done
        curl -sf http://127.0.0.1:30000/health
        # leave teardown margin under the 4h cap: stop sampling at ~3h30
        timeout 12600 python $CODE/sao_plugin/filter_campaign.py \
           --candidates $CANDIDATES --results $RESULTS \
           --server http://127.0.0.1:30000 --model qwen3.5-4b \
           --n ${FILTER_N:-8} --max-tokens ${FILTER_MAX_TOKENS:-24576} \
           --problem-concurrency ${FILTER_CONCURRENCY:-48} || true
        kill \$SERVER_PID 2>/dev/null || true"

# progress report (host side)
wc -l "$RESULTS" "$CANDIDATES" || true
