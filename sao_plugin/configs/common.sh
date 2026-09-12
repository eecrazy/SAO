#!/bin/bash
# paper_006_sao — shared arm config (sourced by sao.sh / grpo_dis.sh AFTER they set ARM).
# Defines the arg arrays consumed by jobs/launch_train.sh. Scope constants:
# paper/experiment_spec.md; component mapping: scratch_impl/implementation_plan.md.

SB=${SB:-/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao}
CODE=${CODE:-$SB/scratch_impl/code}
SLIME_DIR=${SLIME_DIR:-$CODE/slime}
RUNS=${RUNS:-/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/sao_scratch}

export HF_HOME=${HF_HOME:-/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface}
HF_MODEL=${HF_MODEL:-$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
TORCH_DIST=${TORCH_DIST:-$RUNS/qwen3.5-4B_torch_dist}
DATA=${DATA:-$RUNS/data/pool.jsonl}

RUN_DIR=${SAO_RUN_DIR_ROOT:-$RUNS/runs}/$ARM
mkdir -p "$RUN_DIR"

# MODEL_ARGS from the pinned slime clone (dense gated-attention Qwen3.5-4B spec).
source "$SLIME_DIR/scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "$HF_MODEL"
   --ref-load "$TORCH_DIST"
   --load "$RUN_DIR/ckpt"
   --save "$RUN_DIR/ckpt"
   --save-interval "${SAVE_INTERVAL:-10}"
)

ROLLOUT_COMMON=(
   --prompt-data "$DATA"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --custom-rm-path sao_plugin.reward_math.sao_math_rm
   --num-rollout "${NUM_ROLLOUT:-1000}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN:-24576}"
   --rollout-temperature 1.0
   --rollout-top-p 1.0
   --global-batch-size 128
   --balance-data
   # SAO systems contribution: streaming fully-async rollouts + staleness metrics
   # (sao_plugin/async_staleness.py wraps slime's fully_async worker).
   --rollout-function-path sao_plugin.async_staleness.generate_rollout_sao
   --update-weights-interval 1
)

# DIS (method_spec §A): both arms. eps flags reused as the DIS mask bounds (A4).
DIS_ARGS=(
   --use-rollout-logprobs
   --loss-type custom_loss
   --custom-loss-function-path sao_plugin.dis_loss.dis_policy_loss_function
   --eps-clip 0.3
   --eps-clip-high 5.0
   --kl-coef 0.00
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# TP configurable (INT-004: 2-node actor uses TP4 to shard the 24k-token activation
# across 4 ranks → a single long trajectory fits). USE_DIST_OPT=1 adds the
# distributed optimizer (shards optimizer state across DP; recommended multi-node,
# slime docs qwen3-30B) — set for the 2-node runs.
DIST_OPT_ARGS=()
[[ "${USE_DIST_OPT:-0}" == "1" ]] && DIST_OPT_ARGS=(--use-distributed-optimizer)
PERF_ARGS=(
   --tensor-model-parallel-size "${TP:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP:-1}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
   ${DIST_OPT_ARGS[@]+"${DIST_OPT_ARGS[@]}"}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # 0.85 static fraction = larger KV pool (matches the filter/eval servers that
   # ran cleanly at 24k). At 0.75 + 64 concurrent 24k seqs the engines OOM'd
   # (D9, shakedown 5253014: KV alone ~49GB/engine → 73.7GB → CUDA OOM → 503 storm
   # → 0 steps). Qwen3.5-4B has only 8 full-attn layers with growing KV (~32KB/tok),
   # so KV/engine ≈ concurrency × 24k × 32KB.
   --sglang-mem-fraction-static 0.85
   # Bound in-flight long generations so the KV pool can't overflow: 24/engine × 4
   # = 96 in-flight (KV ~18GB/engine, generous headroom) while still exceeding the
   # 128-traj/step consumption via completion turnover. Staleness stays bounded (A13).
   --sglang-server-concurrency "${SGLANG_CONCURRENCY:-24}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project repro-paper006-sao
      --wandb-group "$ARM"
      --wandb-key "$WANDB_API_KEY"
   )
fi
