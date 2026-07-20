"""Staleness-observing wrapper around slime's fully-async rollout — paper_006
method_spec §D / assumption A13.

slime's `generate_rollout_fully_async` streams trajectories from a background
worker but neither reports nor bounds weight-version staleness. SAO's spec says
engines "may undergo multiple updates during a single trajectory" with a
"controlled degree of off-policy bias" (bound unstated). We therefore:

  (a) ALWAYS log per-drain staleness stats computed from Sample.weight_versions
      (stamped by slime's weight-sync backends as stringified ints) —
      `SAO_STALENESS rollout=<id> n=<groups> mean=<m> p50=<p> max=<M> dropped=<d>`
      — consumed by smoke gate S3 and monitoring;
  (b) ENFORCE the pre-registered cap (A13, default 4): groups whose oldest
      generation weight-version lags the newest version observed so far by more
      than SAO_MAX_WEIGHT_STALENESS are DROPPED — discarded outright, never
      requeued. Requeueing COMPLETED samples is unsafe in this slime
      (verification finding D2): generation would continue past the natural EOS
      with a fresh max_new_tokens budget while the stale reward is never
      recomputed, and the group's weight_versions never improve → a
      data-corrupting requeue livelock. Dropping loses only that prompt's visit
      this epoch (it reshuffles back later); at the configured in-flight bound
      (--sglang-server-concurrency × engines ≈ 2 steps of lookahead) drops
      should be rare stragglers. SAO_MAX_WEIGHT_STALENESS=0 = log-only.

Wiring: --rollout-function-path sao_plugin.async_staleness.generate_rollout_sao
(instead of slime.rollout.fully_async_rollout.generate_rollout_fully_async).
"""

import asyncio
import logging
import os
import time

from slime.rollout.fully_async_rollout import _get_global_worker
from slime.utils.async_utils import run
from slime.utils.types import Sample

logger = logging.getLogger("sao_plugin.async_staleness")

_latest_version_seen: int = 0


def _group_versions(group: list[Sample]) -> list[int]:
    versions: list[int] = []
    for sample in group:
        for v in sample.weight_versions:
            try:
                versions.append(int(v))
            except (TypeError, ValueError):
                pass
    return versions


async def _generate_rollout_sao(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    global _latest_version_seen
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    max_stale = int(os.environ.get("SAO_MAX_WEIGHT_STALENESS", "0"))

    collected: dict[int, list[Sample]] = {}
    stalenesses: list[int] = []
    dropped = 0
    started = time.time()
    last_log = started

    while len(collected) < target:
        drained = 0
        for gid, group in worker.get_completed_groups():
            drained += 1
            versions = _group_versions(group)
            if versions:
                _latest_version_seen = max(_latest_version_seen, max(versions))
                staleness = _latest_version_seen - min(versions)
            else:
                # No version stamps (e.g. no weight sync happened yet) → treat as fresh.
                staleness = 0

            if max_stale > 0 and staleness > max_stale:
                # DROP, never requeue: completed samples cannot be safely
                # regenerated in this slime (post-EOS continuation + stale
                # reward + livelock — see module docstring, finding D2).
                dropped += 1
                logger.info(
                    "sao rollout %d: dropped stale group (staleness=%d > cap %d)",
                    rollout_id,
                    staleness,
                    max_stale,
                )
                continue

            collected[gid] = group
            stalenesses.append(staleness)

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > 30.0:
            logger.info(
                "sao rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    if stalenesses:
        ordered = sorted(stalenesses)
        mean = sum(ordered) / len(ordered)
        p50 = ordered[len(ordered) // 2]
        mx = ordered[-1]
    else:
        mean = p50 = mx = 0
    # Fixed-shape line: parsed by smoke gate S3 and the monitoring notes.
    logger.info(
        "SAO_STALENESS rollout=%d n=%d mean=%.2f p50=%d max=%d dropped=%d latest_version=%d",
        rollout_id,
        len(stalenesses),
        mean,
        p50,
        mx,
        dropped,
        _latest_version_seen,
    )

    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    return sorted(collected.values(), key=_key)[:target]


def generate_rollout_sao(args, rollout_id, data_buffer, evaluation: bool = False):
    """slime ``--rollout-function-path`` entrypoint (fully-async + staleness)."""
    if evaluation:
        raise ValueError("sao fully-async rollout doesn't support evaluation mode — eval offline")
    return run(_generate_rollout_sao(args, rollout_id, data_buffer))
