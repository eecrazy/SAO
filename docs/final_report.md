# Final Report — paper_006_sao

A careful reproduction status, not a verdict on the paper. Do not overclaim.

- **outcome:** R5-official-fail
- **date_completed:** 2026-07-19

## Summary

SAO ("Single-Rollout Asynchronous Optimization") claims that, in the same async RL framework at a
matched 128-trajectory/step budget, **single-rollout (n=1) + a learned value model (SAO) outperforms
group-sampling GRPO-w/DIS** by ~+2.7pp on a 3-benchmark math average, both stable where vanilla GRPO
collapses. No official code was ever released (arXiv abs, web search, and a symbol-grep of the
authors' slime framework all negative — `official_run` declared evidence-only failed, INT-001), so
the reproduction ceiling is R5 and everything below is a **from-scratch** independent implementation.

We reproduced Phase-1 (single-turn math, no tools) on slime: Qwen3.5-4B instruct policy+critic, a
Skywork-OR1 difficulty-filtered pool (3948 prompts, base-pass∈[1/8,7/8]@24k), 24k responses, ~500
policy steps (INT-005 halved the horizon from 1000 for budget; the paper says the SAO-vs-GRPO gap
grows after ~400 steps). Both arms share the identical DIS loss path and async regime; the only
intended difference is single-rollout+critic vs 16×8 group sampling. The full SAO algorithm was
implemented and independently fidelity-reviewed (DIS double-sided zero-mask from rollout logprobs,
adaptive-λ token GAE, TTUR K=2, frozen-attention critic verified to freeze the hybrid model's
GatedDeltaNet mixers, value-pretraining warmup).

**Result (avg_pass1_3bench, full n): base 25.26 → GRPO-w/DIS 48.66, SAO 49.82; gain SAO−GRPO = +1.16pp
(paper +2.7).** Match rule PASSES: (i) positivity — SAO>GRPO, both ≫ base; (ii) magnitude —
|1.16−2.7|=1.54 ≤ 2.0pp; (iii) stability — both ran to ~499 steps with reward stable ~0.9, no
collapse; L014 transfer gate — both arms lifted ~+23pp over base. So the paper's central claim
**reproduces directionally** on this 4B setup. Most likely reasons the gain (+1.16) undershoots the
paper's (+2.7): the halved horizon (mid-training @220 steps showed +2.5pp on 2 benches, i.e. the gap
was developing), statistical noise (see caveats), and 4B-vs-30B scale. No implementation-bug
attribution — all components validated end-to-end.

## Result table

See [`result_table.md`](result_table.md). Gain: paper +2.7 · scratch **+1.16** (SE 1.78, 95% CI
[−2.30, +4.63]). Per-bench: AIME25 −1.67, BeyondAIME +4.56, HMMT +0.57 (SAO−GRPO).

## Human Supervision Summary

<!-- generated: supervision_summary -->

| stage | n_interactions | n_blocking | n_directional | top_type | n_avoidable | n_generative | open |
|---|---:|---:|---:|---|---:|---:|---:|
| scoping | 3 | 0 | 3 | experimental_design | 0 | 2 | 0 |
| official_run | 0 | 0 | 0 | - | 0 | 0 | 0 |
| scratch_impl | 2 | 2 | 0 | resource_provision | 0 | 0 | 0 |
| repair | 0 | 0 | 0 | - | 0 | 0 | 0 |
| analysis | 0 | 0 | 0 | - | 0 | 0 | 0 |
| **total** | 5 | 2 | 3 | - | 0 | 2 | 0 |

<!-- /generated -->

Full interaction log: [`../human_interactions/interactions.md`](../human_interactions/interactions.md)

INT-001 scope approval (R5 ceiling); INT-002 slime base; INT-003 Qwen3.5-4B + harder Skywork-OR1 data
(generative — user-proposed model+data direction); INT-004 2-node + raised budget (blocking, after the
8-GPU 24k memory/throughput wall); INT-005 24k×500 horizon (blocking, after the measured cost + length
runaway). The two blocking scratch_impl INTs were genuine cost/scope owner-decisions the agent surfaced
with measured evidence but could not self-authorize.

## Lessons exported to meta

- L017 — probe the authors' own open framework for method symbols at scoping (slime greps clean → generic infra, not a de-facto release).
- L018 — adversarial pre-GPU fidelity review as a standard gate (found 6 pre-GPU defects + the filter format-cache trap).
- L019 — evaluating RL on recent benchmarks with a newer base model: pretraining contamination (2025 benches in a 2026 model) inflates absolute levels; rely on the PAIRED gain, not absolute pass@1.
- L020 — never compute eval metrics from an incrementally-written results file mid-run (spurious HMMT 95.9% from a partial JSONL; complete → 52.2%). Confirm the eval job COMPLETED first.
- F007 — fully-async partial-rollout continuation re-dispatches with a FRESH full max_new_tokens → response length accumulates past the cap (24k→36k) → actor OOM + runaway cost. Fix: continuation budget = max_response_len − already_generated.
- F008 — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is incompatible with slime's TorchMemorySaver (crashes init); control training memory via max-tokens-per-gpu / TP instead.

## Caveats

- **Single seed, 500 (not 1000) steps, 4B (not 30B), single-turn (no tools).** Direction reproduces;
  the gain is NOT statistically significant (95% CI includes 0) — this is a directional, not decisive,
  reproduction.
- **Contamination:** AIME-2025/HMMT-Feb-2025 predate the 2026 model's cutoff → absolute post-RL levels
  likely inflated (equally for both arms; paired gain valid). Pool itself is contamination-clean.
- Compute: ~715 (SAO) + ~560 (GRPO) + ~155 filter + ~90 evals ≈ 1520 GPU-h (2-node, INT-004 raised cap).
- HMMT-Feb25 substituted for the paper's HMMT-Nov25 (logged); absolute paper-vs-scratch not compared (scale).
