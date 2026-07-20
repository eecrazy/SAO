# Mismatch Analysis — paper_006_sao

The scratch reproduction meets the pre-registered match rule (gain +1.16 vs paper +2.7, within
±2.0pp; positivity + stability pass). There is NO official code, so this is a paper↔scratch analysis,
not a scratch↔official diff. Focus: (a) why the *gain magnitude* is smaller than the paper's, and
(b) strength-of-evidence caveats. Absolute levels are on a different scale (4B single-turn vs 30B-MoE
TIR) and are not directly compared.

## Scratch vs paper — setting diff (adaptations, all pre-registered in experiment_spec/INTs)

| dimension | paper | scratch | matters? |
|---|---|---|---|
| model | Qwen3-30B-A3B-Thinking (MoE), 128k | Qwen3.5-4B instruct (dense hybrid), 24k | scale ↓ → absolute levels not comparable; gain is the readout (INT-003) |
| task | TIR math (Python tool) + SWE coding | single-turn math, no tools | Phase-1 scope (INT-001); tool-use out of scope |
| RL data | UNSTATED in paper | Skywork-OR1 filtered, base-pass∈[1/8,7/8]@24k (3948) | adaptation, not deviation (paper data unstated); pool clean of eval overlap |
| horizon | ~1000 steps | ~500 steps (INT-005, budget) | fewer steps → gain may be under-developed (paper divergence grows after ~400) |
| algorithm (DIS, single-rollout GAE, TTUR, frozen-attn, value-pretrain) | as specified | faithfully implemented + fidelity-reviewed (wf_a0d05795) | matched — the intended variable |
| eval | AIME25/BeyondAIME/HMMT + IMO | AIME25/BeyondAIME/HMMT-Feb25 (HMMT-Nov→Feb deviation) | metric parity; boxed grading train==eval |

## Gap attribution (gain +1.16 scratch vs +2.7 paper)

1. **Compute budget / horizon (confidence: MED-HIGH).** 500 vs ~1000 steps. The paper states the
   SAO-vs-GRPO divergence *grows* after ~400 steps; the mid-training curve @220 steps already showed
   +2.5pp (2-bench), and the 500-step 3-bench final is +1.16pp — consistent with a gain that is real
   but still developing and noise-dominated at this eval size. More steps likely widen it.
2. **Statistical power (confidence: HIGH).** Paired bootstrap gain SE = 1.78pp; 95% CI [−2.30, +4.63]
   includes 0. The +1.16 point estimate reproduces the paper's direction and magnitude-within-tolerance,
   but is NOT significantly distinguishable from 0 at n=64/32/64, single seed. Per-bench: SAO wins
   BeyondAIME (+4.56) and HMMT (+0.57) but loses AIME25 (−1.67) — the gain rides on BeyondAIME.
3. **Scale (confidence: MED).** SAO's critic/value machinery may pay off more on the 30B MoE than on a
   4B; the gain could genuinely be smaller at 4B.
4. **NOT attributable to implementation bugs.** All algorithm components validated (SAO_FREEZE 44–60%
   frozen incl. GDN mixers; TTUR K=2; DIS mask sane; adaptive-λ; staleness ≤4; LR continuity L013;
   F001 conversion lossless). Fidelity review found only the (fixed) filter-cache trap; DIS clean.

## Residual unknowns / caveats

- **Contamination (important).** Qwen3.5-4B is a 2026 model; AIME-2025 and HMMT-Feb-2025 predate its
  cutoff. Base→post-RL jumps (25→50 avg3) are likely inflated by pretraining exposure surfaced by RL
  (per-problem spread argues against pure memorization, but absolute levels are suspect). Affects
  **both arms equally** → the *paired gain* stays valid; absolute pass@1 is not comparable to the
  paper's 30B numbers. The training pool is clean (0 exact/fragment overlap with eval benches).
- **Single seed, 500 steps, 4B.** Direction reproduces; a decisive claim would need more seeds, the
  full ~1000-step horizon, and a contamination-controlled eval (a post-2026-cutoff benchmark).
- **HMMT-Feb25 vs paper's HMMT-Nov25** (logged deviation) and 4B-vs-30B scale bound cross-comparison.
