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

# paper_006_sao — pass@1 eval segment (resumable; chain with singleton until all
# samples land). Serves an HF checkpoint via sglang (dp=8) and runs
# eval/run_eval.py over the requested benchmarks.
# Env: MODEL_DIR (HF ckpt dir; converted offline per F001), TAG (results name),
#      BENCHES (default "aime25 beyondaime hmmt25 math500"),
#      N_AIME/N_BEYOND/N_HMMT/N_MATH500 sample counts.
set -euxo pipefail

CONTAINER=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/slime_nightly-dev-20260707a.sqsh
SB=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao
CODE=$SB/scratch_impl/code
export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface
RUNS=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/sao_scratch
BENCH_DIR=${BENCH_DIR:-$RUNS/data/bench}
OUT_DIR=$RUNS/eval/${TAG:?set TAG}
mkdir -p "$OUT_DIR"

: "${MODEL_DIR:?set MODEL_DIR (HF checkpoint dir)}"
BENCHES=${BENCHES:-"aime25 beyondaime hmmt25 math500"}
N_AIME=${N_AIME:-64}; N_BEYOND=${N_BEYOND:-32}; N_HMMT=${N_HMMT:-64}; N_MATH500=${N_MATH500:-8}

srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home \
     --no-container-entrypoint \
     bash -c "set -euxo pipefail
        export HF_HOME=$HF_HOME
        export PYTHONPATH=$CODE
        pkill -9 sglang 2>/dev/null || true; sleep 2
        python -m sglang.launch_server \
           --model-path $MODEL_DIR --served-model-name sao-eval \
           --dp-size 8 --tp-size 1 --mem-fraction-static 0.85 \
           --host 127.0.0.1 --port 30000 --log-level warning &
        SPID=\$!
        for i in \$(seq 120); do curl -sf http://127.0.0.1:30000/health >/dev/null && break; sleep 5; done
        curl -sf http://127.0.0.1:30000/health
        deadline=\$((SECONDS + 12600))   # stop sampling ~3h30 in, leave teardown margin
        for bench in $BENCHES; do
           case \$bench in
             aime25) N=$N_AIME;; beyondaime) N=$N_BEYOND;; hmmt25) N=$N_HMMT;; math500) N=$N_MATH500;; *) N=8;;
           esac
           left=\$((deadline - SECONDS)); [ \$left -le 60 ] && break
           timeout \$left python $CODE/eval/run_eval.py sample \
              --bench $BENCH_DIR/\$bench.jsonl \
              --results $OUT_DIR/\$bench.jsonl \
              --server http://127.0.0.1:30000 --model sao-eval \
              --n \$N --max-tokens ${EVAL_MAX_TOKENS:-32768} || true
        done
        kill \$SPID 2>/dev/null || true
        python $CODE/eval/run_eval.py aggregate --results $OUT_DIR/*.jsonl || true"
