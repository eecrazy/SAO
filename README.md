# SAO — Independent Reproduction

An **independent, from-scratch reproduction** of

> **Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning** (SAO)
> Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong — Tsinghua / Z.AI — arXiv:2607.07508

> ⚠️ **This is NOT the authors' official code.** The SAO paper released no code, data, or
> checkpoints. Everything here is a clean-room re-implementation built on top of
> [THUDM/slime](https://github.com/THUDM/slime) from the paper's description only, plus an
> analysis of whether the paper's central claim reproduces. All credit for the method belongs to
> the original authors; any errors here are the reproduction's, not theirs.

## What SAO claims

In the same asynchronous RL framework, at a **matched 128-trajectory/step budget**, replacing
group-sampling GRPO with **single-rollout (n=1) sampling + a learned value model (SAO)** improves a
3-benchmark math average by **≈ +2.7pp** over GRPO-with-DIS, with both stable where vanilla GRPO
collapses. SAO's ingredients:

- **DIS** — double-sided *zero-masked* importance sampling using rollout-engine logprobs (no `π_old`), token-level, bounds `ε_ℓ=0.3 / ε_h=5.0`.
- **Single-rollout + critic** — n=1 with a learned critic and length-adaptive token-level GAE (`λ_policy = 1 − 1/(1.5·len)`, `λ_critic = 1`), TTUR (K=2 critic updates/step), **frozen-attention critic**, and a critic-only value-pretraining warmup.
- **Async streaming rollouts** — the learner consumes 128-trajectory batches as they arrive; engines may be several weight-versions stale, and DIS handles the off-policy correction.

## Reproduction result (this repo)

Phase-1 setting: single-turn math RL (no tools), **Qwen3.5-4B** instruct policy+critic on slime,
a Skywork-OR1 difficulty-filtered pool (3,948 prompts, base pass-rate ∈ [1/8, 7/8] @ 24k), 24k-token
responses, **~500 policy steps** (halved from the paper's ~1000 for compute budget), 2 nodes / 16×H100.

Primary metric `avg_pass1_3bench` (mean pass@1 over AIME2025 n=64, BeyondAIME n=32, HMMT-Feb25 n=64):

| method | base | **GRPO-w/DIS** | **SAO** |
|---|---:|---:|---:|
| avg_pass1_3bench | 25.26 | 48.66 | **49.82** |

**Gain (SAO − GRPO-w/DIS) = +1.16pp** (paper reports +2.7pp on a 30B MoE); paired bootstrap SE 1.78pp.

**Verdict — the central claim reproduces directionally** under a pre-registered match rule
(SAO > GRPO paired ∧ both ≫ base ∧ |gain − 2.7| ≤ 2.0pp ∧ stability). Honest caveats:

- The gain is **not statistically significant** (95% CI [−2.30, +4.63] includes 0) — directional, not decisive. SAO wins BeyondAIME (+4.6) and HMMT (+0.6) but loses AIME25 (−1.7).
- The smaller-than-paper magnitude is most consistent with the **halved horizon** (mid-training @220 steps already showed +2.5pp on 2 benches, still growing), noise, and 4B-vs-30B scale — not any implementation bug (all components were validated + fidelity-reviewed).
- **Contamination**: AIME-2025 / HMMT-Feb-2025 predate the 2026 base model's cutoff, so absolute post-RL levels are likely inflated (equally for both arms → the *paired gain* is the valid readout; the training pool itself is contamination-clean).

Full analysis: [`docs/final_report.md`](docs/final_report.md), [`docs/result_table.md`](docs/result_table.md),
[`docs/mismatch_analysis.md`](docs/mismatch_analysis.md). Frozen method spec (paper-only):
[`docs/method_spec.md`](docs/method_spec.md).

### Completed 8×H20 run

The later single-node `h20_v1` run used the committed 3,993-prompt pool. Its
three-benchmark mean was 25.125 base, 47.389 SAO, and 47.934 GRPO-w/DIS, so SAO
trailed GRPO-w/DIS by 0.545 pp in this run. This is a separate experiment from
the H100 table above. Full configuration, per-benchmark scores, and caveats are
in [`docs/H20_RESULTS.md`](docs/H20_RESULTS.md).

## Repo layout

```
sao_plugin/           SAO implementation (slime plugin — additive)
  dis_loss.py           DIS double-sided zero-mask policy loss (--custom-loss-function-path)
  adaptive_gae.py       length-adaptive token GAE, actor-only (--custom-advantage-function-path)
  async_staleness.py    staleness-observing/​-bounding wrapper over slime's fully-async rollout
  reward_math.py        boxed-answer math grader (train == eval == filter)
  prompt.py             shared "put your final answer in \boxed{}" instruction
  data_prep.py          Skywork-OR1 → difficulty-banded candidate pool
  filter_campaign.py    base-pass-rate filter (keeps prompts in [1/8, 7/8]); format-aware resume cache
  configs/              common.sh + sao.sh + grpo_dis.sh (arm arg arrays)
eval/
  prep_benchmarks.py    materialize AIME25 / BeyondAIME / HMMT-Feb25 / MATH500 (instructed prompts)
  run_eval.py           resumable pass@1 scorer (same grader as training)
jobs/                   SLURM launchers (single-node + 2-node multi-node ray; filter; convert; eval)
patches/
  slime_sao.patch       in-place slime edits: TTUR, frozen-attention log, warmup-save,
                        and the fully-async response-length hard-cap (see below)
docs/                   method_spec, experiment_spec, and the reproduction analysis
reproducibility/        dataset/source revisions, hardware, results, and SHA256 manifest
requirements/           complete hash-locked Python 3.12/cu130 runtime resolution
```

> Paths in `jobs/*.sbatch|sh` are **cluster-specific** (they hard-code a `/lustre/...` workspace and
> an enroot container image). Adjust `CONTAINER`, `SB`, `RUNS`, `HF_HOME` and the account/partition
> to your environment. Secrets (`WANDB_API_KEY`, `HF_TOKEN`) are sourced from a git-ignored `.env`,
> never hard-coded.

## Setup

### Local 8×H20 setup with uv

For the current host, use the checked-in host-native workflow instead of the
cluster-specific SLURM/Enroot launchers:

```bash
./scripts/pipeline.sh setup
./scripts/pipeline.sh preflight
./scripts/pipeline.sh download
./scripts/pipeline.sh data
./scripts/pipeline.sh filter
./scripts/pipeline.sh convert
./scripts/pipeline.sh train sao smoke
./scripts/pipeline.sh train grpo_dis smoke
./scripts/pipeline.sh ready
```

It creates `.venv` with uv, pins the CUDA/RL stack used by the reference image,
applies the slime/SAO/Megatron/SGLang patches, and configures 4 H20s for the
actor+critic plus 4 for rollouts. Full training, export, evaluation, resume
semantics, and hardware rationale are documented in
[`docs/LOCAL_UV.md`](docs/LOCAL_UV.md).

The exact training/evaluation JSONL snapshots are committed; model exports and
checkpoints are intentionally not. After cloning, verify all snapshot hashes,
row coverage, and reported aggregates with:

```bash
python scripts/verify_reproducibility.py
```

See [`reproducibility/environment.md`](reproducibility/environment.md) for the
dependency-lock provenance and the precise limits on bitwise reproducibility.

### Original container/cluster setup

1. **slime** — clone and pin, then apply the patch:
   ```bash
   git clone https://github.com/THUDM/slime && cd slime
   git checkout 680824dd5e01a2e83750bf87fc366ec6fa98766c
   git apply /path/to/this/repo/patches/slime_sao.patch
   ```
   Container used: `slimerl/slime:nightly-dev-20260707a` (Megatron-Core + SGLang + Ray).
2. **Put `sao_plugin/` and `eval/` on `PYTHONPATH`** alongside slime (the configs reference
   `sao_plugin.*` / `eval.*` module paths).
3. **Model**: `Qwen/Qwen3.5-4B` (instruct). Convert HF → Megatron torch_dist once
   (`jobs/prep_megatron_ckpt.sbatch`) for the policy+critic init.
4. **Secrets**: create a `.env` exporting `WANDB_API_KEY` and `HF_TOKEN` (auto-sourced by the launchers).

## Running the experiments

**1. Build the training pool** (Skywork-OR1 → difficulty band → base-pass-rate filter):
```bash
# candidate pool (stratified over the easier difficulty tiers, WITH the boxing instruction)
python sao_plugin/data_prep.py --out data/candidates.jsonl --stratified --per-tier 2000 --d32-min 0 --d32-max 4
# eval benchmarks (same instructed prompts as training)
python eval/prep_benchmarks.py --out-dir data/bench
# base-pass-rate filter (serves Qwen3.5-4B via sglang, n=8 @ 24k) — resumable singleton chain
sbatch jobs/filter.sbatch     # repeat/chain until data/filter_results.jsonl covers the candidates
python sao_plugin/filter_campaign.py --candidates data/candidates.jsonl \
       --results data/filter_results.jsonl --emit data/pool.jsonl   # keeps base-pass ∈ [1/8, 7/8]
```

**2. Base anchor eval** (both arms must beat this):
```bash
sbatch --export=ALL,MODEL_DIR=<hf_base>,TAG=base jobs/eval.sbatch
```

**3. Train both arms** (2 nodes / 16 GPU; actor+critic TP4×DP2 on node 0, 8 SGLang engines on node 1).
Each is a resumable 4h singleton chain — a 24k×500-step run is ~12 segments per arm:
```bash
ENV="ACTOR_GPUS=8,ROLLOUT_GPUS=8,TP=4,USE_DIST_OPT=1,MAX_TOKENS_PER_GPU=32768,SGLANG_CONCURRENCY=32,DATA=data/pool.jsonl,RUN_TAG=v1"
for i in $(seq 14); do sbatch --job-name=sao-tr-sao  --dependency=singleton \
   --export=ALL,ARM_CONFIG=sao_plugin/configs/sao.sh,$ENV      jobs/submit_train_2node.sbatch; done
for i in $(seq 12); do sbatch --job-name=sao-tr-grpo --dependency=singleton \
   --export=ALL,ARM_CONFIG=sao_plugin/configs/grpo_dis.sh,$ENV jobs/submit_train_2node.sbatch; done
```
SAO = `--advantage-estimator ppo --rollout-batch-size 128 --n-samples-per-prompt 1` + critic + TTUR +
adaptive-λ + frozen-attention; GRPO-w/DIS = `--advantage-estimator grpo --rollout-batch-size 16
--n-samples-per-prompt 8` (std-norm off, no critic). Both share the identical DIS loss and async regime.

**4. Final eval + comparison** (convert final checkpoints → HF → full-n 3-bench):
```bash
sbatch --export=ALL,ITER_DIR=<run>/ckpt/iter_XXXXXXX,OUT_DIR=<hf_out> jobs/convert_hf.sbatch
sbatch --export=ALL,MODEL_DIR=<hf_out>,TAG=final_sao,BENCHES="aime25 beyondaime hmmt25 math500",N_AIME=64,N_BEYOND=32,N_HMMT=64 jobs/eval.sbatch
python eval/run_eval.py aggregate --results <out>/*.jsonl   # -> avg_pass1_3bench
```

## Implementation notes (things that mattered)

- **Frozen-attention on a hybrid model.** Qwen3.5-4B interleaves full-attention and GatedDeltaNet
  (linear-attention) layers; slime nests both mixer types under the `self_attention` submodule, so
  the freeze regex `["self_attention"]` correctly freezes *all* token-mixers (verified: ~44–60% of
  params frozen, incl. `linear_attn.{in_proj_qkv,conv1d,A_log,dt_bias}`).
- **Response-length hard cap (patch).** slime's fully-async rollout aborts in-flight generations on
  weight updates and requeues them; without a fix the continuation gets a *fresh* full `max_new_tokens`,
  so a trajectory spanning K weight-versions accumulates up to K× the cap (observed 24k → 36k → actor
  OOM). The patch caps the continuation budget to `max_response_len − already_generated`.
- **Answer format == everywhere.** The grader is boxed-only; every prompt the model sees (filter,
  training, eval) carries the same boxing instruction, or reward is silently ~0.

## License / attribution

Method © the original SAO authors (arXiv:2607.07508). This reproduction builds on
[THUDM/slime](https://github.com/THUDM/slime) (see its license). Reproduction code here is provided
as-is for research transparency.
