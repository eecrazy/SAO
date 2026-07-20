"""Shared answer-format instruction for paper_006_sao.

The rule-based reward (reward_math.grade_response) and slime's extract_answer
are BOXED-ONLY: a response with no `\\boxed{...}` in its post-</think> segment
grades 0, no matter how correct the prose answer is. Base Qwen3.5-4B, given a
bare math question, rarely emits `\\boxed{}` on its own — a calibration probe
(job 5225102) measured <1% answer-presence UNIFORMLY across every difficulty
tier (d32 0..8), including problems trivial for the 32B teacher. That is a
format artifact, not a difficulty signal.

So every prompt the model ever sees — filter campaign, RL training rollouts, and
eval — must carry the same explicit boxing instruction, or (a) training gets ~0
reward on everything and never learns, and (b) train/eval formats diverge. This
module is the single source of that instruction; data_prep.py (train pool +
calibration) and eval/prep_benchmarks.py (eval sets) both import it.
"""

BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def with_instruction(question: str) -> str:
    """Append the boxing instruction to a bare question (idempotent)."""
    q = question.rstrip()
    if "\\boxed" in q and "put your final answer" in q:
        return q  # already instructed
    return f"{q}\n\n{BOXED_INSTRUCTION}"
