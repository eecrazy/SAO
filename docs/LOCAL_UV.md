# Local uv training guide (8 x NVIDIA H20)

This host-native profile replaces the original cluster-only SLURM/Enroot paths.
It keeps the upstream experiment definition and pins the same source revisions:

- SAO repository: `98c02713a650bb1f390d661b478ad3ef95e66cb0`
- slime: `680824dd5e01a2e83750bf87fc366ec6fa98766c` plus `patches/slime_sao.patch`
- Megatron-LM: `1dcf0dafa884ad52ffb243625717a3471643e087` plus slime's Megatron patch
- SGLang: `0.5.13` plus all four slime SGLang patches
- Python 3.12, PyTorch 2.11.0 + CUDA 13.0 ABI, Transformer Engine 2.16.1

The reference container is cu129, but this host exposes driver 580 and SGLang
0.5.13's published wheels target torch 2.11/cu130. The local profile therefore
uses slime's supported CUDA-13 build path. Native extensions compile against the
consistent CUDA prefix installed inside `.venv`, while the host CUDA 13.3
toolkit is only a bootstrap fallback. This is a systems-only adaptation.

## Hardware profile

The detected machine has 8 NVIDIA H20 GPUs, 97,871 MiB each, SM90, with NV18
between every GPU pair. The local defaults are therefore:

| role | GPUs | parallelism / memory setting |
|---|---:|---|
| actor + critic | 4 | TP=2, DP=2, recompute, 12,288 tokens/GPU for SAO |
| rollout | 4 | four TP=1 SGLang engines, concurrency 24 each, static memory 0.85 |
| GRPO actor | 4 | TP=2, DP=2, 32,768 tokens/GPU (no resident critic) |

This matches the repository's validated single-node topology. H20 has more
memory than the H100 profile, but the SAO token cap remains conservative because
the actor and critic are colocated and the 24k response tail is the OOM risk.

## Staged workflow

Run from the repository root. Every long stage is resumable and deliberately
separate; full filtering and two 500-step arms consume substantial GPU time.

```bash
cp .env.example .env                 # optional: HF_TOKEN / WANDB_API_KEY
./scripts/pipeline.sh setup
./scripts/pipeline.sh preflight
./scripts/pipeline.sh download
./scripts/pipeline.sh data
./scripts/pipeline.sh filter
./scripts/pipeline.sh convert
```

The filter samples each candidate 8 times at a 24,576-token cap and appends to
`data/filter_results.jsonl`. Re-run the stage after interruption; completed UIDs
are skipped. It emits `data/pool.jsonl` and its stats file only after every
candidate has a complete, error-free 8-sample result.

The completed local campaign graded all 10,000 candidates and retained 3,993
prompts in the inclusive 1/8–7/8 difficulty band. Its base-pass histogram for
pass counts 0 through 8 is `4109, 969, 604, 511, 443, 399, 456, 611, 1898`.

Before a full run, execute both three-rollout smoke gates:

```bash
./scripts/pipeline.sh train sao smoke
./scripts/pipeline.sh train grpo_dis smoke
./scripts/pipeline.sh ready
```

The local path has been exercised end to end on the stated 8×H20 host. The SAO
gate completed one critic-only warmup plus two actor updates, including two
critic updates per actor step, frozen critic attention, adaptive GAE, DIS,
staleness tracking, weight synchronization, and iteration-2 actor/critic
checkpoints. The GRPO-with-DIS gate completed two actor updates and its
iteration-2 checkpoint. Both actor checkpoints were converted back to complete
two-shard HF checkpoints, and a short-output MATH500 run verified HF loading,
SGLang serving, resumable sampling, grading, aggregation, and server cleanup.
That short-output run is an integration gate only; its scores are not benchmark
results.

The `ready` gate is intentionally strict: it only succeeds when filtering has
graded every candidate, the emitted pool is atomic, unique, difficulty-banded,
and prompt-format consistent, both smoke checkpoint series reached iteration 2,
the full SAO/GRPO configurations retain their registered horizons and batch
geometry, and at least 10 TiB remains for the two checkpoint series (about
8.95 TiB at the validated smoke-checkpoint sizes).

Full training uses the README horizons (SAO: 47 critic-only warmup + 500 policy
updates; GRPO-with-DIS: 500 policy updates). Re-running the same arm resumes from
its latest torch_dist checkpoint. Set `RUN_TAG` before starting a distinct run.

```bash
RUN_TAG=h20_v1 ./scripts/pipeline.sh train sao full
RUN_TAG=h20_v1 ./scripts/pipeline.sh train grpo_dis full
```

Export and evaluate the final actor checkpoints:

```bash
./scripts/pipeline.sh export runs/sao_h20_v1/ckpt/iter_XXXXXXX artifacts/final_sao_h20_v1
./scripts/pipeline.sh export runs/grpo_dis_h20_v1/ckpt/iter_XXXXXXX artifacts/final_grpo_h20_v1
./scripts/pipeline.sh eval artifacts/models/Qwen3.5-4B base
./scripts/pipeline.sh eval artifacts/final_sao_h20_v1 final_sao
./scripts/pipeline.sh eval artifacts/final_grpo_h20_v1 final_grpo
```

Evaluation is also resumable per `(problem UID, sample index)`. Results are under
`runs/eval/<tag>/`; `eval/run_eval.py aggregate` prints the three-benchmark mean.

## Useful overrides

All launch settings are environment overrides. Common safe controls include
`MAX_RESPONSE_LEN`, `SGLANG_CONCURRENCY`, `SAO_MAX_TOKENS_PER_GPU`,
`GRPO_MAX_TOKENS_PER_GPU`, `SAVE_INTERVAL`, `NUM_ROLLOUT`, and `RUN_TAG`.
Changing batch geometry, sample count, DIS bounds, GAE, reward formatting, or
arm-specific algorithms changes the experiment and should not be treated as a
hardware adjustment.

The setup defaults to FlashAttention 2, which is sufficient on H20. Set
`INSTALL_FA3=1` when running `setup_uv.sh` to compile the image's optional FA3
extension as well; it is an optimization, not an SAO algorithm requirement.
