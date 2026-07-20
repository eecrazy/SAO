# Scratch Implementation Results — paper_006_sao

Pre-repair numbers only (repair results go in `../reference_guided_repair/results.md`).
Every number cites job ID(s) + log path.

## Baseline

| date | job_id(s) | command_ref | metric_value | log_path | notes |
|---|---|---|---:|---|---|

- **final_value:** -
- **seeds:** -

## Proposed Method

| date | job_id(s) | command_ref | metric_value | log_path | notes |
|---|---|---|---:|---|---|

- **final_value:** -
- **seeds:** -

## Comparison vs paper and official run

| method | paper | official | scratch | gap_vs_paper_pct | gap_vs_official_pct | within match_rule? |
|---|---:|---:|---:|---:|---:|---|
| baseline | - | - | - | - | - | - |
| proposed | - | - | - | - | - | - |

## Production run (2-node, 24k×500, RUN_TAG=v1) — IN PROGRESS

- **F001 export probe PASSED** (job 5273925 convert iter_39 → 5273926 eval): MATH500 **88.05%** vs
  base 89.41% → HF conversion lossless, eval pipeline validated. (AIME25 48.6% is small-n noise:
  partial 22/30 problems @ n=16 vs base n=64.) iter_39 is within the 47-step critic-only warmup so
  the actor ≈ base, as expected.
- **SAO arm LEARNING** (5269573+, wandb sao_v1): pool reward rose 0.30→0.49→0.53→0.65→0.70→0.76 over
  the first 14 policy steps; response length capped ~23k (D-len fix holding), no OOM. Cost ~1.38 GPU-h/step.
- **L013/F006 LR-continuity PASS**: across SAO segment-1→2 resume, actor LR steady 1e-6, critic 5e-6
  (no reset/freeze); reward stayed ~0.8 (trained ckpt loaded, not base) → slime resume correct.
- **GRPO arm (5269587+) also LEARNING** (wandb grpo_dis_v1): raw_reward 0.63→0.90 (note `rollout/rewards`
  is the group-mean-CENTERED advantage ≈0 by construction — read `raw_reward` for the actual grade).
  Both arms shorten responses as they learn (SAO 23k→12k, GRPO 17k→5k). Both converge ~0.9 pool reward;
  the real test is held-out AIME/HMMT transfer (curve + final evals). GRPO faster (no critic) → ahead
  on iters (219 vs SAO 99).
- NEXT: curve evals every ~50 steps (convert → eval reduced-n AIME25); LR continuity check at each
  segment resume (L013/F006); final base(25.26)/SAO/GRPO @ full n at step 500 → match rule.

## Curve eval @ ~220 policy steps (matched) — first held-out transfer signal

Jobs 5284302 (SAO iter_269) / 5284304 (GRPO iter_219), full n=32, complete (960 samples/bench).

| bench | base | SAO@220 | GRPO@220 | SAO−GRPO |
|---|---:|---:|---:|---:|
| AIME25 | 37.4 | **66.1** | 62.1 | +4.0 |
| HMMT25 | 23.4 | **53.2** | 52.2 | +1.0 |
| 2-bench avg | 30.4 | **59.7** | 57.2 | **+2.5** |

- **SAO > GRPO on BOTH benchmarks** at matched training → 2-bench-avg gain **+2.5pp**, strikingly close
  to the paper's +2.7pp (mid-training, 2-of-3 benches, n=32 — not the final metric, but right direction+magnitude).
- Per-problem distribution is a natural difficulty spread (SAO HMMT: 7 solved / 16 partial / 7 hard),
  NOT uniform ~100% → genuine solving, not memorization.
- **⚠ CONTAMINATION CAVEAT**: Qwen3.5-4B is a 2026 model; AIME-2025/HMMT-Feb-2025 predate its cutoff, so
  absolute post-RL levels (base 37/23 → 66/53) are likely inflated by pretraining exposure surfaced by RL.
  This affects BOTH arms equally → the PAIRED gain (SAO−GRPO) remains the valid readout; absolute pass@1
  is not comparable to the paper's 30B numbers. Pool itself is clean (0 exact/fragment overlap w/ benches).
- **⚠ LESSON**: initial watcher read HMMT 95.9% from an INCOMPLETELY-WRITTEN eval file (incremental JSONL);
  always confirm the eval job COMPLETED before computing metrics. Complete file → 52.2%.

## Base anchor (G1) — Qwen3.5-4B instruct, no RL

Job 5227855 (`sao-eval-base`, TAG=base), avg_pass1_3bench protocol (temp 1.0, top_p 1.0, 32k),
prompts carry the boxing instruction (train==eval). Results `…/eval/base/*.jsonl`.

| benchmark | n | problems | base pass@1 |
|---|---:|---:|---:|
| AIME2025 | 64 | 30 | 37.45% |
| BeyondAIME | 32 | 100 | 14.94% |
| HMMT-Feb25 | 64 | 30 | 23.39% |
| **avg_pass1_3bench** | | | **25.26%** |
| MATH500 (anchor, partial n) | ~5 | 339 | 89.41% |

Validates: (1) **eval harness is sound** — MATH500 at 89% (a broken grader/boxing/sglang path
would read ~0%), the G1 harness-sanity gate (L012); (2) base 4B is **non-saturated** on the hard
benchmarks → RL headroom, and a floor both arms must beat. **Match-rule base anchor = 25.26%**
(both SAO and GRPO-w/DIS final must exceed this). Paper's 84.1/86.8 are on a 30B MoE →
scale-mismatched, used only for the |gain−2.7|≤2.0pp magnitude test, not the anchor.

## Data calibration (base Qwen3.5-4B pass-rate vs difficulty)

Job 5227261 (`sao-calib`, n=8 @ 24k, WITH boxing instruction; results
`data/calib_results_v2.jsonl`), stratified 48/tier over Skywork-OR1 d32 (= R1-Distill-32B
fails/16). Cross-check: job 5225102 (same set WITHOUT instruction) → <1% answer-presence
uniformly (the F-fmt boxing bug; see implementation_plan.md).

| d32 | mean pass@1 | in-band [1/8,7/8] | 0/8 | 8/8 | answer% | trunc% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.68 | 42% | 15% | 44% | 71% | 30% |
| 1 | 0.41 | 40% | 35% | 25% | 47% | 55% |
| 2 | 0.32 | 38% | 44% | 19% | 42% | 61% |
| 3 | 0.22 | 29% | 62% | 8% | 39% | 65% |
| 4 | 0.32 | 40% | 46% | 15% | 40% | 62% |
| 5 | 0.15 | 31% | 65% | 4% | 21% | 81% |
| 6 | 0.18 | 29% | 67% | 4% | 31% | 73% |

Findings: (1) once boxing is fixed, d32 IS predictive of 4B difficulty (pass declines with d32);
(2) the 4B is strong but NOT saturated even at d32=0 (mean 0.68, 42% in-band) — a genuinely hard
pool for it; (3) in-band yield is ~30–42% across ALL tiers (as pass drops, mass flows 8/8 →
in-band → 0/8); (4) truncation at 24k rises with d32 (30%→81%) — higher tiers waste more rollout
budget, and the [1/8,7/8] filter self-selects problems solvable within 24k.

**Band decision: d32 ∈ [0,6], stratified** — best in-band yield with the lowest truncation.
Projected ~35% yield → ~4.2k pool from ~12k candidates (≈30 epochs at 128 traj/step × 1000 steps;
acceptable because (a) both arms share the pool so size is not a comparison confound, and (b)
eval is on held-out AIME/HMMT so training-prompt reuse can't inflate the metric). Widen to [0,8]
or raise candidate count if the emitted pool < 3k. NEXT: regen candidates.jsonl (stratified [0,6],
WITH instruction) → full filter → `filter_campaign.py --emit pool.jsonl`.

## Issues
