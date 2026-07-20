"""SAO length-adaptive token-level GAE — paper_006 method_spec.md §B.2 (VAPO).

Per-sequence lambda_policy = 1 - 1/(alpha * response_len), alpha = 1.5 (paper);
gamma from --gamma (1.0, assumption A3). Replicates the "ppo" branch of
slime's compute_advantages_and_returns with a per-row lambda vector —
`vanilla_gae` broadcasts a [B] lambda cleanly, so no slime patch is needed.

Wiring — ACTOR ROLE ONLY (shared CLI):
    --advantage-estimator ppo
    --custom-advantage-function-path sao_plugin.adaptive_gae.sao_adaptive_gae
The CRITIC role must override `custom_advantage_function_path: null` in the
--megatron-config-path YAML so its value-loss targets come from the stock PPO
path with scalar --lambd 1.0 → lambda_critic = 1 exactly (method_spec B.2).

alpha override: env SAO_GAE_LAMBDA_ALPHA (default 1.5).
"""

import os
from argparse import Namespace

import torch
from megatron.core import mpu

from slime.utils.ppo_utils import get_advantages_and_returns_batch
from slime.utils.types import RolloutBatch


def sao_adaptive_gae(args: Namespace, rollout_data: RolloutBatch) -> None:
    values = rollout_data.get("values")
    assert values is not None, (
        "sao_adaptive_gae needs critic values — use --advantage-estimator ppo and make sure "
        "the critic role is running (train_critic ships values to the actor)."
    )
    kl = rollout_data["kl"]  # zeros when kl_coef == 0 (our setting)
    scalar_rewards = rollout_data.get("rewards")
    response_lengths = rollout_data.get("response_lengths")
    total_lengths = rollout_data.get("total_lengths")

    alpha = float(os.environ.get("SAO_GAE_LAMBDA_ALPHA", "1.5"))

    # Token-level rewards: -kl_coef * kl per token, terminal task reward added at
    # the last response token. Mirrors slime's stock ppo branch (loss.py) incl.
    # the CP detail that cp_rank 0 holds the sequence tail under zigzag sharding;
    # done out-of-place so rollout_data["kl"] stays pristine for logging.
    kl_coef = -args.kl_coef
    cp_rank = mpu.get_context_parallel_rank()
    token_rewards = []
    for reward, k in zip(scalar_rewards, kl, strict=False):
        k = k * kl_coef
        if cp_rank == 0 and k.numel() > 0:
            k[-1] = k[-1] + reward
        token_rewards.append(k)

    lambd_vec = torch.tensor(
        [1.0 - 1.0 / (alpha * max(int(length), 1)) for length in response_lengths],
        dtype=values[0].dtype,
        device=values[0].device,
    )

    # chunked=False: chunked_gae builds a scalar-lambda kernel matrix and cannot
    # take a per-row lambda; vanilla_gae broadcasts [B] fine. O(T) loop measured
    # at gate S2 (fallback: extend chunked_gae with per-row kernels).
    advantages, returns = get_advantages_and_returns_batch(
        total_lengths,
        response_lengths,
        values,
        token_rewards,
        args.gamma,
        lambd_vec,
        chunked=False,
    )
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
