# Result Table — paper_006_sao

Schema and gap definitions: `meta/schema.md`. Scope mirror of `../paper/experiment_spec.md`.

- **metric_name:** avg_pass1_3bench (mean pass@1 over AIME2025 n=64, BeyondAIME n=32, HMMT-Feb25 n=64)
- **metric_unit:** percent
- **higher_is_better:** yes
- **setting:** single-turn math RL (no tools) on slime @680824dd, 2-node (actor+critic 8 GPU TP4×DP2, rollout 8 GPU); Qwen3.5-4B instruct policy+critic; Skywork-OR1 filtered pool (3948 prompts, base-pass∈[1/8,7/8]@24k); 24k train / 32k eval; ~500 policy steps (INT-005; paper ~1000); temp 1.0, top_p 1.0, KL/entropy 0
- **paper_table_ref:** paper math-TIR table, 3-bench avg (AIME25+BeyondAIME+HMMT): SAO 86.8 / GRPO-w/DIS 84.1 → gain +2.7 (on 30B-A3B MoE)
- **match_rule:** custom — (i) SAO>GRPO-w/DIS paired ∧ both>base; (ii) |gain−2.7|≤2.0pp (escape ±1 bootstrap SE); (iii) stability. Judged on the **gain** (scale-robust); paper absolute values are 30B-scale, not comparable to our 4B.
- **match_rule_note:** REPRODUCES — (i) PASS (49.82>48.66, both≫25.26), (ii) PASS (|1.16−2.7|=1.54≤2.0), (iii) PASS (both ~499 steps, stable ~0.9 reward, no collapse). Official code never released → scratch judged vs paper → **R5-official-fail**. Gain not statistically significant (95% CI [−2.30,+4.63]); see mismatch_analysis.md.

| row_id | source | method | seeds | value | std | gap_paper | gap_paper_pct | gap_official_pct | gpu_h | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| paper/baseline | paper | baseline | 1 | 84.1 | - | - | - | - | - | reported, 30B-A3B MoE (scale-mismatched) |
| official/baseline | official | baseline | - | - | - | - | - | - | - | NO official code (evidence-only fail, INT-001) |
| scratch/baseline | scratch | baseline | 1 | 48.66 | 1.28 | n/a¹ | n/a¹ | - | ~560 | GRPO-w/DIS, iter_499; base anchor 25.26 |
| paper/proposed | paper | proposed | 1 | 86.8 | - | - | - | - | - | reported, 30B-A3B MoE |
| official/proposed | official | proposed | - | - | - | - | - | - | - | NO official code |
| scratch/proposed | scratch | proposed | 1 | 49.82 | 1.24 | n/a¹ | n/a¹ | - | ~715 | SAO, iter_546 |
| repair/proposed | repair | proposed | - | - | - | - | - | - | - | not entered (scratch reproduced) |

**gain (SAO − GRPO-w/DIS): paper +2.7 · scratch +1.16 (SE 1.78, 95% CI [−2.30,+4.63]) · |Δgain|=1.54 ≤2.0 → PASS**

¹ Paper values are 30B-A3B-MoE TIR (Python tool); our 4B single-turn no-tool absolute scale differs, so
paper-vs-scratch *absolute* gap is not meaningful. The match rule targets the scale-robust **gain**.

## Per-benchmark pass@1 (scratch, full n) — see mismatch_analysis.md

| benchmark | n | base | SAO | GRPO-w/DIS | SAO−GRPO |
|---|---:|---:|---:|---:|---:|
| AIME2025 | 64 | 37.45 | 58.07 | 59.74 | −1.67 |
| BeyondAIME | 32 | 14.94 | 42.62 | 38.06 | +4.56 |
| HMMT-Feb25 | 64 | 23.39 | 48.75 | 48.18 | +0.57 |
| **avg3** | | **25.26** | **49.82** | **48.66** | **+1.16** |
| MATH500 (anchor L012) | 8 | 89.41 | — | — | — |

Jobs: base 5227855; SAO 5269573–86 → eval 5304604/05; GRPO 5269587–98 → eval 5304607/08.
wandb `repro-paper006-sao` (`sao_v1`, `grpo_dis_v1`).
