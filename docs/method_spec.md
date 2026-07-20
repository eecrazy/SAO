# Method Spec — paper_006_sao

**Written from the paper ONLY — no official-code peeking.** This is the implementation
contract for `scratch_impl`. Mark every detail the paper does not state as
`[not stated in paper]` — those markings become the assumption list in
`scratch_impl/implementation_plan.md` and the prime suspects during `repair`.

Frozen at scoping exit (2026-07-12). Source: arXiv 2607.07508v1 LaTeX
(`2.preliminary.tex`, `4.method.tex`, `5.experiment.tex`, `7.appendix.tex`).

SAO has two separable contributions. Both are in reproduction scope; the agentic
multi-turn structure (contribution B3) is exercised only in the pre-registered
Phase-2 extension.

## Inputs / outputs

- **Input:** a prompt dataset D of verifiable problems; a policy LLM π_θ; a value
  model V_φ (same backbone + scalar head); a rollout inference engine that returns,
  for every generated token, the **log-prob under the weights that generated it**
  (π_rollout).
- **Output:** trained π_θ (and V_φ). Per prompt, exactly ONE trajectory is sampled
  per visit (single-rollout); trajectories carry (tokens, per-token rollout
  log-probs, reward, per-token weight-version info [implied by the async design]).

## Algorithm (loss / objective / procedure, step by step)

### A. DIS — Direct double-sided Importance Sampling (token-level, both arms)

1. Behavior proxy: drop π_old entirely; use the rollout engine's logged log-probs.
   Per token t: `r_t(θ) = exp(log π_θ(a_t|s_t) − log π_rollout(a_t|s_t))`.
2. Calibration (zero-mask, NOT clip-to-boundary):
   `f(x; ε_l, ε_h) = x if 1−ε_l < x < 1+ε_h else 0`.
3. Objective as printed: `L(θ) = Ê_t[ f(r_t) · Â_t · log π_θ(a_t|s_t) ]`.
   Note (derivation, not paper text): for in-region tokens
   ∇_θ[r_t·Â_t] = Â_t·r_t·∇log π_θ, identical to the gradient of the printed
   objective with f(r_t) treated as the IS coefficient; out-of-region tokens
   contribute exactly zero gradient. Implementation as a masked unclipped surrogate
   `−mask_t · r_t · Â_t` is therefore gradient-equivalent.
   [not stated in paper]: whether f(r_t) is detached; whether masked tokens are
   removed from the loss denominator (we keep them in).
4. No KL penalty and no entropy bonus appear in the stated objective.
   [not stated in paper]: explicit confirmation KL/entropy coefficients are 0.

### B. Single-rollout with a value model (SAO arm)

1. **Sampling:** one rollout per prompt (group size 1). Batch = 128 trajectories
   per training step.
2. **Advantage:** token-level GAE with critic V_φ:
   - length-adaptive λ_policy = 1 − 1/(α·l) with α = 1.5 (VAPO), where l = response
     length [not stated in paper: whether l counts response tokens only — assumed —
     or full sequence];
   - λ_critic = 1 (critic's return targets);
   - γ [not stated in paper] — assume 1.0.
3. **Value model:**
   - init from the same SFT model as the policy (both "initialized from" the
     finetuned model);
   - **value pretraining** on a 6k-sample corpus before RL (ablation: 3.2k);
     construction/targets/epochs [not stated in paper];
   - **faster value update (TTUR):** K = 2 value-model gradient updates per 1 policy
     update per batch;
   - **frozen-attention:** during RL, attention-module parameters of V_φ frozen;
     MoE (FFN) projections optimized. On a dense/hybrid backbone
     [not stated in paper]: analog = freeze all token-mixer modules, train
     MLP/FFN + value head (+ norms/embeddings [not stated in paper] — we train them);
   - critic LR 5e-6, 10-step warmup; value loss [not stated in paper: exact form] —
     assume standard clipped value loss / MSE to λ_critic=1 returns.
4. **Skip-observation GAE (multi-turn only):** for trajectory
   T = [a_0, o_0, a_1, o_1, …], advantages propagate across action boundaries,
   bypassing observation tokens:
   `Â(a_{i,N}) = δ + γλ·Â(a_{i+1,0})`, `δ = r_t + γ·V(a_{i+1,0}) − V(a_{i,N})`.
   Values are neither trained on nor queried at observation tokens. In a single-turn
   setting there are no observation tokens and this reduces to standard token GAE
   (Phase-1 note). Step-level value variants were tried and rejected by the paper
   (appendix): token-level wins.

### C. Baseline arm — GRPO w/ DIS (same async framework)

- 16 prompts × 8 rollouts = 128 trajectories/step; group-mean advantage
  (GRPO; std normalization [not stated in paper] — the preliminaries present
  mean/std normalization; the headline "GRPO" baseline is "standard GRPO with
  clip-higher"; for the DIS variant we assume the same group normalization with
  DIS replacing the PPO clipping);
- identical DIS token masking, identical batch/LR/length settings;
- no critic.
- (Context anchors from the paper, not implemented: vanilla GRPO clip-higher
  collapses ~step 160; vanilla VAPO collapses ~90; running-mean single-rollout
  baseline with window 8 underperforms badly.)

### D. Asynchronous training loop (both arms)

- Rollout engines generate continuously; each completed trajectory becomes
  available to the learner immediately (no group-waiting — with group size 1 there
  is no intra-group straggler by construction).
- The learner consumes batches of 128 trajectories per optimizer step.
- Rollout engines may run weights several updates old; a single trajectory may span
  multiple weight versions ("rollout engines may undergo multiple updates during a
  single trajectory generation") — this is why π_old tracking is dropped and DIS
  corrects against π_rollout.
- Staleness bound [not stated in paper] — "we accept a controlled degree of
  off-policy bias". Weight-sync cadence, partial-rollout handling
  [not stated in paper].

## Hyperparameters stated in the paper

- policy LR: 1e-6
- value LR: 5e-6; value warmup: 10 steps
- batch size: 128 trajectories/step; SAO group size 1; GRPO variants 16 prompts × 8
- max length: 128k tokens (paper scale)
- DIS math: ε_low = 0.3, ε_high = 5.0 (coding: 0.8 / 3.0)
- GAE: λ_policy = 1 − 1/(α·l), α = 1.5; λ_critic = 1
- TTUR: K = 2 value updates per batch
- value pretraining corpus: 6k samples (ablation 3.2k)
- training horizon: ~1000 steps (curves; "train stably for one thousand steps")
- SFT init: Qwen3-30B-A3B-Thinking-2507, 3 epochs on GPT-OSS-120B TIR data
- [not stated in paper]: γ, optimizer family/betas/weight-decay, grad-clip, LR
  schedule shape, KL/entropy coefficients (implied 0), rollout temperature during
  training (eval uses 1.0), staleness bound, weight-sync cadence, value-loss form,
  value-pretrain recipe, RL prompt dataset, reward function details (implied
  binary verifiable-answer reward for math [not stated in paper]).

## Training procedure

1. SFT the base model (paper: TIR data; our Phase-1 analog: none — use the released
   instruct model as the "post-SFT" init; adaptation logged in implementation_plan).
2. Initialize π_θ and V_φ from the SFT model; V_φ gets a scalar value head
   [not stated in paper: head init] — assume fresh/random.
3. Value pretraining (paper: 6k-sample corpus; our analog: critic-only warmup steps
   on on-policy rollouts, ≈6k trajectories, policy frozen).
4. Async RL loop per step: collect 128 trajectories (streaming, staleness-bounded);
   compute rewards; critic forward → token values; skip-observation length-adaptive
   GAE → advantages; K=2 critic updates (λ_critic=1 returns); 1 policy update with
   DIS-masked objective; push fresh weights to rollout engines
   [cadence not stated in paper].
5. Horizon ~1000 steps; checkpoint cadence [not stated in paper].

## Evaluation procedure

- Metric: pass@1 accuracy per benchmark; report mean over 16 eval runs (AIME2025,
  HMMT, IMOAnswerBench) or 4 runs (BeyondAIME).
- Decoding: temperature 1.0, top_p 1.0, max generation 128k tokens; math allows up
  to 50 turns (TIR tool calls); SWE-Bench 300 OpenHands turns.
- Benchmarks (math): AIME2025, BeyondAIME, HMMT Nov 2025, IMOAnswerBench.
- Answer checking [not stated in paper] — assume standard boxed-answer exact-match
  verification for math.
- Our Phase-1 mapping (spec'd in experiment_spec.md): single-turn no-tools, 32k eval
  length, n≥32 samples/problem pass@1 mean, AIME2025 + BeyondAIME + HMMT-Feb-2025
  (version deviation logged), deepscaler-style grader.
