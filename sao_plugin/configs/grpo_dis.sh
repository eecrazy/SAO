#!/bin/bash
# paper_006_sao — baseline arm: GRPO w/ DIS in the SAME async regime.
# 16 prompts × 8 rollouts = 128 traj/step, group-mean advantage (std-norm OFF,
# assumption A15), no critic. method_spec §C; implementation_plan C1.

ARM=grpo_dis${RUN_TAG:+_${RUN_TAG}}
CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# INT-004b: 500 policy steps (24k×500). GRPO has no critic-only warmup, so
# NUM_ROLLOUT == policy steps == 500 (matches the SAO arm's 500 policy updates).
NUM_ROLLOUT=${NUM_ROLLOUT:-500}
source "$CONFIG_DIR/common.sh"

# Pre-registered staleness cap (A13): over-cap groups are DROPPED (see
# sao_plugin/async_staleness.py; same value in both arms for comparability).
export SAO_MAX_WEIGHT_STALENESS=${SAO_MAX_WEIGHT_STALENESS:-4}

ALGO_ARGS=(
   --advantage-estimator grpo
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --disable-grpo-std-normalization
   --gamma 1.0
)

# 8-GPU split: actor 4 + rollout 4 (rollout GPUs equal across arms so the
# staleness distributions are comparable).
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
