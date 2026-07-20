"""SAO DIS policy loss — paper_006 (arXiv 2607.07508 §4.1), method_spec.md §A.

Direct double-sided Importance Sampling: the IS ratio is computed against the
ROLLOUT ENGINE's logged per-token log-probs (pi_old is dropped entirely — no
old-policy forward pass), and tokens outside the trust region are ZERO-MASKED
out of the gradient (not clipped to the boundary):

    r_t = exp(log pi_theta(a_t|s_t) - log pi_rollout(a_t|s_t))
    L_t = - 1[1 - eps_l < r_t < 1 + eps_h] * r_t * A_t

Gradient-equivalent to the paper's printed objective L = f(r_t) A_t log pi_theta
for in-region tokens (d r / d theta = r * d log pi / d theta); out-of-region
tokens contribute exactly zero (method_spec ambiguity #9). Masked tokens stay in
the loss denominator (assumption A10): `sum_of_sample_mean` is built from the
unmodified loss_masks.

Wiring (both arms):
    --loss-type custom_loss
    --custom-loss-function-path sao_plugin.dis_loss.dis_policy_loss_function
    --use-rollout-logprobs            # makes slime skip the pi_old recompute
    --eps-clip 0.3 --eps-clip-high 5.0   # reused as the DIS bounds (A4)
"""

from argparse import Namespace
from typing import Callable

import torch

from slime.backends.megatron_utils.loss import (
    get_log_probs_and_entropy,
    get_rollout_top_p_logprob_kwargs,
)
from slime.utils.types import RolloutBatch


def dis_policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # SAO preconditions — fail loudly rather than silently diverge from the spec.
    assert args.use_rollout_logprobs, "SAO DIS requires --use-rollout-logprobs (r_t vs pi_rollout, no pi_old)"
    assert batch.get("rollout_log_probs"), "rollout_log_probs missing — sglang must return logprobs"
    assert not args.use_tis, "SAO DIS replaces TIS; do not enable --use-tis"
    assert not getattr(args, "use_opsm", False), "OPSM not part of the SAO spec"
    assert not args.use_kl_loss and args.kl_coef == 0, "SAO objective has no KL term (method_spec §A.4)"
    assert args.advantage_estimator != "gspo", "sequence-level GSPO incompatible with token-level DIS"

    advantages = torch.cat(batch["advantages"], dim=0)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
        **get_rollout_top_p_logprob_kwargs(args, batch),
    )
    log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)
    rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)

    # ppo_kl kept in slime's sign convention (old - new) for metric comparability.
    ppo_kl = rollout_log_probs - log_probs
    ratio = (-ppo_kl).exp()

    in_region = (ratio > 1.0 - args.eps_clip) & (ratio < 1.0 + args.eps_clip_high)
    # torch.where (not indicator-multiply) so a masked out-of-region token can never
    # produce inf*0 = NaN if a ratio ever overflows — value-identical in-region, but a
    # single NaN would poison the whole 1000-step run (fidelity review wf_a0d05795 note).
    pg_loss = torch.where(in_region, -(ratio * advantages), torch.zeros_like(ratio))
    dis_masked = 1.0 - in_region.to(ratio.dtype)

    pg_loss = sum_of_sample_mean(pg_loss)

    entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    # Keep autograd traversing the full graph even when this CP rank holds no
    # loss-contributing tokens (same guard as slime's stock policy loss).
    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    reported = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        # Paper Fig-3c analog: fraction of tokens zero-masked by DIS.
        "dis_mask_frac": sum_of_sample_mean(dis_masked.detach()),
        "dis_ratio_mean": sum_of_sample_mean(ratio.detach()),
        "ppo_kl": sum_of_sample_mean(ppo_kl.detach()),
        "train_rollout_logprob_abs_diff": sum_of_sample_mean((log_probs.detach() - rollout_log_probs).abs()),
    }
    return loss, reported
