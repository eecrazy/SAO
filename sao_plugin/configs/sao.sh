#!/bin/bash
# paper_006_sao — SAO arm (proposed method): single-rollout (n=1) + critic +
# adaptive-λ token GAE + DIS + TTUR K=2 + frozen-attention critic + critic-only
# value pretraining. method_spec §B; implementation_plan C1–C6.

ARM=sao${RUN_TAG:+_${RUN_TAG}}
CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Horizon: 47 critic-only warmup rollouts happen INSIDE num_rollout
# (train_async.py gates actor training on rollout_id >= num_critic_only_steps),
# so the sao arm runs 1047 total → exactly 1000 POLICY updates, matching the
# grpo_dis arm's 1000 (verification finding D3). Must be set BEFORE common.sh
# builds ROLLOUT_COMMON.
# INT-004b: horizon cut to 500 policy steps (24k×500, budget) → 47 critic-only
# warmup + 500 policy = 547. Paper's SAO-vs-GRPO divergence appears by ~400 steps.
NUM_ROLLOUT=${NUM_ROLLOUT:-547}

# The SAO arm colocates the critic on the actor's GPUs, so per-GPU peak memory is
# actor(train) + critic(resident) on one 80GB card. At max-tokens-per-gpu=32768
# this OOMs (D8: actor peak ~76GB + 8.4GB critic > 80GB). The activation peak
# scales ~linearly with max-tokens (32768→~52GB peak), so a 12288 cap drops it to
# ~19GB, leaving ~generous headroom under the ~70GB usable (80 − critic − TMS
# margin). This changes only gradient-accum granularity, not the effective batch
# or the algorithm; GRPO arm (no critic) keeps 32768.
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — it is INCOMPATIBLE
# with slime's TorchMemorySaver (critic/rollout offload) and crashes init (D8b,
# smoke 5224754). Memory is controlled purely via this token cap.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}

source "$CONFIG_DIR/common.sh"

# SAO knobs consumed by sao_plugin + the TTUR patch (exported into the ray
# runtime env by jobs/launch_train.sh).
export SAO_CRITIC_UPDATES_PER_STEP=2   # TTUR K=2 (method_spec B.3)
export SAO_GAE_LAMBDA_ALPHA=1.5        # λ_policy = 1 − 1/(1.5·len) (B.2)
# Pre-registered staleness cap (A13, experiment_spec regime): groups older than
# 4 weight-versions are DROPPED (not requeued — requeueing completed samples
# corrupts them in this slime; verification finding D2), enforced in
# sao_plugin/async_staleness.py.
export SAO_MAX_WEIGHT_STALENESS=${SAO_MAX_WEIGHT_STALENESS:-4}

# Per-role overrides: critic LR 5e-6 + 10-step warmup (method_spec B.3), critic
# ckpt dirs, stock scalar-λ PPO returns for the critic (λ_critic = 1 — the
# custom adaptive-λ advantage fn applies to the ACTOR only), frozen-attention
# regexes (A8; pinned/verified at gate S0 via the trainable-param dump).
#
# Critic load bootstrap (verification finding D4): slime's missing-checkpoint
# fallback (load→ref_load) runs on the base args BEFORE per-role YAML overrides,
# and load_checkpoint hard-asserts the dir exists — so a static
# `load: critic_ckpt` crashes every fresh launch. The YAML is regenerated at
# each segment start, so pick per segment: resume from critic_ckpt once it has
# a checkpoint, else initialize from the base torch_dist.
if [[ -f "$RUN_DIR/critic_ckpt/latest_checkpointed_iteration.txt" ]]; then
   CRITIC_LOAD=$RUN_DIR/critic_ckpt
else
   CRITIC_LOAD=$TORCH_DIST
fi
# Clamp critic LR warmup to < NUM_ROLLOUT so Megatron's
# assert(lr_warmup_steps < lr_decay_steps) holds even in SMOKE (NUM_ROLLOUT=3).
CRITIC_LR_WARMUP=$(( NUM_ROLLOUT > 10 ? 10 : 0 ))
ROLES_YAML=$RUN_DIR/megatron_roles_sao.yaml
cat > "$ROLES_YAML" << EOF
megatron:
  - name: default
    role: critic
    overrides:
      lr: 5.0e-6
      lr_warmup_iters: $CRITIC_LR_WARMUP
      load: $CRITIC_LOAD
      save: $RUN_DIR/critic_ckpt
      custom_advantage_function_path: null
      freeze_params_name_list: ["self_attention"]
EOF

ALGO_ARGS=(
   --advantage-estimator ppo
   --rollout-batch-size 128
   --n-samples-per-prompt 1
   --custom-advantage-function-path sao_plugin.adaptive_gae.sao_adaptive_gae
   --gamma 1.0
   --lambd 1.0
   --value-clip 0.2
   --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS:-47}"
   --megatron-config-path "$ROLES_YAML"
)

# 8-GPU split (verification finding D6): slime colocates the critic on the SAME
# bundles as the actor (offload-swapped), so the split is actor+critic on 4
# GPUs + rollout on 4 GPUs — all 8 used, symmetric with the grpo_dis arm.
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
