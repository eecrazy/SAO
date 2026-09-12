# H20 experiment configuration and results

This page records the completed local `h20_v1` experiment. It is independent of
the earlier 2-node 16×H100 result in `final_report.md`: the machine, filtered
pool (3,993 versus 3,948 prompts), checkpoints, and scores differ. Do not merge
the two result tables.

## Configuration

| item | H20 setting |
|---|---|
| hardware | 1 node, 8× NVIDIA H20 97,871 MiB, SM90, all-to-all NV18 |
| model | Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| data | committed `data/pool.jsonl`, 3,993 prompts, SHA256-verified |
| split | actor (+ critic for SAO) 4 GPUs; four 1-GPU rollout engines |
| parallelism | TP=2, DP=2, PP=1, CP=1; bf16; full recomputation |
| rollout | 24,576 max response tokens, temperature=1.0, top_p=1.0 |
| async | weight update interval 1; maximum accepted staleness 4 |
| DIS | zero-masked ratio interval [0.3, 5.0], KL=0, entropy=0 |
| optimizer | Adam, actor LR 1e-6, beta=(0.9, 0.98), weight decay 0.1 |
| rollout serving | 4× TP1 SGLang, concurrency 24/engine, static memory 0.85 |
| SAO | 47 critic-only warmups + 500 actor updates; n=1×128; critic LR 5e-6, 10-step warmup, TTUR K=2, actor max tokens/GPU 12,288 |
| GRPO-w/DIS | 500 actor updates; 16 prompts×8 samples; max tokens/GPU 32,768; group std normalization disabled |

The SAO run started 2026-07-29 and completed 2026-08-02 at actor and critic
checkpoint marker 546 (547 zero-indexed rollout iterations). The GRPO-w/DIS run
started 2026-08-02 and completed 2026-08-03 at marker 499. Both Ray jobs ended
successfully according to their retained local logs; the large logs themselves
are not committed.

## Final evaluation

Scores are problem-macro pass@1 percentages recomputed from the committed
per-sample JSONL files. AIME25/HMMT use 30×64 samples, BeyondAIME 100×32, and
MATH500 500×8. Generation used temperature=1.0, top_p=1.0, a 32,768-token cap,
and no explicit API seed.

| method | AIME25 | BeyondAIME | HMMT25 | 3-bench avg | MATH500 |
|---|---:|---:|---:|---:|---:|
| Base | 37.083 | 14.594 | 23.698 | 25.125 | 88.025 |
| SAO | 58.125 | **41.125** | 42.917 | 47.389 | 95.425 |
| GRPO-w/DIS | **59.323** | 39.844 | **44.635** | **47.934** | **96.175** |
| SAO − GRPO-w/DIS | −1.198 | +1.281 | −1.719 | **−0.545** | −0.750 |

Both RL arms improve substantially over base, but this H20 run does **not**
reproduce SAO's directional advantage over GRPO-w/DIS: SAO trails by 0.545
percentage points on the registered three-benchmark mean. This does not alter
the separately reported H100 run (+1.16 pp), and neither single run establishes
a statistically robust method ranking.

Machine-readable values are in `reproducibility/h20_results.json`; exact sample
coverage and checksums are checked by `scripts/verify_reproducibility.py`.
