# Experiment Spec — paper_006_sao

The scope contract: which single representative setting this reproduction targets and
what counts as success. Scoping exits only when every field below is filled and the
human has approved the scope (logged as an INT-### record, referenced in CLAUDE.md).
**This file is the canonical scope source** — the CLAUDE.md Scope block, the result
table front matter, and the registry fields are mirrors copied from here.

Scope approved in plan mode 2026-07-12 → INT-001. Official_run is evidence-only
failed (no released code) → **outcome ceiling R5** (L001; stated in INT-001).

## Selected setting

**Phase 1 (the match-rule scope): single-turn math RL, no tools, on slime.**
The paper's own setting (Qwen3-30B-A3B, 128k ctx, multi-turn TIR + SWE-agent, ~1000
async steps) is far beyond budget; we reproduce the paper's algorithmic claim —
single-rollout + value model beats group-sampling GRPO w/DIS in the same
asynchronous framework at matched trajectory budget — at small scale in our own
harness:

- **Model:** Qwen/Qwen3.5-4B (Instruct; rev 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
  as BOTH policy and critic init (paper analog: post-SFT init; no extra SFT stage —
  adaptation A-SFT). Hybrid attention (full-attn + gated-deltanet), vocab 248,320.
  User directive (INT-001): "try qwen3.5, start with 4B".
- **Framework:** slime @680824dd (async RL, rollout logprobs, PPO critic native) —
  user directive; miles fallback (see scratch_impl/codebase_selection.md).
- **Train data:** Skywork/Skywork-OR1-RL-Data math split (@1cdedc52) → hard-band
  pre-selection on its difficulty labels → **our base-pass-rate filter** with
  Qwen3.5-4B (n=8 @24k over ~10–15k candidates; keep pass∈[1/8, 7/8]) → target
  ≥8–10k prompts. Filter campaign trajectories are retained (difficulty stats +
  optional offline value pretrain). The paper's RL dataset is unstated, so this is
  an adaptation, not a deviation (notes.md ambiguity #1); "slightly harder than
  DeepScaleR" per INT-001 + L014.
- **Regime (both arms identical):** async streaming rollouts
  (update-weights-interval 1, max-weight-staleness 4, pause-mode in_place),
  128 trajectories/optimizer step, 1000 steps, 24k train response cap, rollout
  temp 1.0, KL coef 0, entropy coef 0, DIS ε_l=0.3 / ε_h=5.0.
- **Phase 2 (pre-registered extension, NOT in the match rule):** port the
  meta_reason v3 MR/E/FA multi-turn setting to slime and apply SAO n=1 vs GRPO n=8
  to cut rollout cost (~8× E-token reduction hypothesis). Own INT + budget
  (~+300 GPU-h placeholder) before any Phase-2 job.

## Chosen baseline

**GRPO w/ DIS in the same async framework**: 16 prompts × 8 rollouts = 128
trajectories/step, group-mean advantage (std normalization off — Dr.GRPO-style,
assumption A-GRPONORM), identical DIS masking, no critic. This is the paper's own
strong baseline (Table 1 row read as "GRPO (w/ DIS)" = 94.2 AIME25; the
fig:main_results comparison curve) and isolates exactly the single-rollout+critic
contribution. Alternatives rejected:
- vanilla GRPO clip-higher (paper: collapses ~step 160) — kept only as an OPTIONAL
  secondary instability-contrast arm (~200–300 steps) if budget remains; not a fair
  headline baseline since it lacks the off-policy correction the async regime needs;
- vanilla VAPO / running-mean — ablation-tier in the paper, weaker baselines;
- synchronous GRPO — would change two variables at once (sampling AND async regime).

## Target metric

- **metric_name:** avg_pass1_3bench
- **metric_unit:** percentage points (accuracy %)
- **higher_is_better:** yes
- **definition:** unweighted mean over three benchmarks of per-benchmark mean pass@1:
  AIME2025 (30 problems, n=64 samples/problem), BeyondAIME (100 problems, n=32),
  HMMT Feb 2025 (30 problems, n=64). Decoding: temperature 1.0, top_p 1.0, max 32k
  tokens, single turn, no tools. Grading: deepscaler-style boxed-answer
  exact-match (math_verify-equivalent rule-based). pass@1 per problem = fraction of
  n samples correct; benchmark score = mean over problems × 100. Arm-vs-arm
  contrasts use PAIRED per-problem statistics; gap-to-paper uses independent SEs
  (L016). MATH500 (n=8) is a saturated harness anchor (L012), not in the metric.
- **paper_table_ref:** Table 1 (`tab:main_results`), arXiv 2607.07508v1 §5

Deviations from the paper's eval protocol (logged): HMMT Feb 2025 instead of
Nov 2025 (Nov set not sourced); IMOAnswerBench omitted (availability unverified;
add as secondary if trivially sourced); 32k eval length instead of 128k; n≥32
samples instead of 16/4 runs; single-turn instead of 50-turn TIR.

## Paper-reported numbers

| method | value | std/seeds (if reported) |
|---|---:|---|
| baseline (GRPO w/ DIS, 3-bench avg of 94.2/71.5/86.7) | 84.1 | not reported |
| proposed (SAO, 3-bench avg of 97.3/74.8/88.3) | 86.8 | not reported |

Paper gain (proposed − baseline) = **+2.7pp** on the 3-bench avg (AIME25-only:
+3.1). Paper absolutes are at Qwen3-30B-A3B/TIR/128k scale — recorded for the
result table but NOT comparable to our 4B single-turn absolutes; the computable
predicate is on OUR gain (below).

## Reproduction tolerance

- **match_rule:** custom
- **match_rule_note:** Gain-based (L002: paper reports no variance; the headline
  effect is a gain; absolutes are scale-mismatched across 30B-MoE→4B). Let
  `gain = SAO_avg_pass1_3bench − GRPOwDIS_avg_pass1_3bench` at the final
  checkpoint, both arms evaluated in the same harness. `scratch/proposed`
  reproduces vs `<ref>` iff BOTH:
  (i) **positivity/ordering:** SAO > GRPO-w/DIS on the paired per-problem contrast
  AND both trained arms > the instruct-base anchor (RL itself worked — L014 guard);
  (ii) **magnitude:** `|gain_ours − gain_ref| ≤ 2.0` pp, with `gain_ref = +2.7`
  (paper row; official_run failed ⇒ paper is the reference per schema).
  Escape hatch (pre-registered): if the paired bootstrap SE of `gain_ours` exceeds
  2.0pp, widen (ii) to ±1 SE and state so (L002/L016).
  **Stability gate (pre-registered, NOT part of the predicate):** both DIS arms
  complete the 1000-step horizon with no collapse — trained-arm eval never drops
  below 50% of its peak gain-over-base and stays there for >100 steps. The
  vanilla-GRPO instability contrast (collapse <400 steps), if run, is a narrative
  check only.

## Compute budget estimate

**Cap: 1500 GPU-h** (INT-001; paper_004 precedent 700–1450). Partition
`batch_block1`, 1 node × 8×H100 per job, 03:55 walltime, `--dependency=singleton`
chains, checkpoint-resumable (save-interval ≤10 steps).

| item | est. GPU-h |
|---|---:|
| S-1 container/infra gate (import + slime PPO CI smoke) | ~10 |
| Difficulty-filter campaign (10–15k prompts × n=8 @24k, sglang) | ~40 |
| Smoke gates S0–S4 + G1 base evals | ~40 |
| Critic-only warmup (47 steps, value pretrain analog) | ~30 |
| `sao` arm 1000 steps @ ~0.6–0.9 GPU-h/step | 600–900 |
| `grpo_dis` arm 1000 steps | 550–800 |
| Curve evals (every 50 steps, AIME25 n=16) + final 3-bench n≥32 ×3 models | ~100 |
| optional `grpo_vanilla` (200–300 steps) | ~30 (only if headroom) |

**S5 go/no-go (pre-registered):** a 30–50-step shakedown of BOTH arms measures real
GPU-h/step BEFORE chains launch; if > 0.75 GPU-h/step, fallback = cut horizon to
600 steps or response cap to 16k, and escalate to the human. LR gate (L013) checked
at every segment boundary; L014 transfer gate at ~step 300 (baseline arm must lift
the eval suite > 1 SE over base, else pause + escalate).

## Pre-registered defaults (adjustable only via a logged INT)

advantage whitening OFF; GRPO std-normalization OFF; γ=1.0; max-weight-staleness 4;
critic warmup = 10-step LR warmup inside 47 critic-only steps; curve-eval cadence
50 steps; eval seeds fixed; repair_attempt_budget 5.

## Scope approval

- **scope_approval:** INT-001 (2026-07-12, plan-mode approval; user-edited options:
  slime primary + miles fallback, Qwen3.5-4B, Skywork-OR1 + filter, positivity +
  |Δgain|≤2.0pp, 1500 GPU-h + S5 go/no-go, 24k/32k lengths).
