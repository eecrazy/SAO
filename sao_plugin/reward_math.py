"""Rule-based math reward for paper_006 — deepscaler grading WITHOUT the
</think> gate.

slime's stock `--rm-type deepscaler` returns 0 for any response that lacks a
"</think>" or "###Response" marker (rm_hub/deepscaler.py). paper_004 hit exactly
this with an instruct model that emits no think tags — every reward silently 0.
Qwen3.5-4B's emission style is a runtime property, so we grade robustly: prefer
the post-</think> segment when present, otherwise grade the full response.
Binary 0/1 exact match via slime's own mathd/sympy graders (assumption A11).

Wiring: --custom-rm-path sao_plugin.reward_math.sao_math_rm
(async signature per slime.rollout.rm_hub.async_rm).

The same `grade_response` is imported by the eval scorer and the filter
campaign so training reward == eval grading == filter grading.
"""

from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def grade_response(response: str, label) -> int:
    if not response or label is None or label == "":
        return 0
    solution = response.split("</think>")[-1] if "</think>" in response else response
    model_answer = extract_answer(solution)
    if model_answer is None:
        return 0

    truth = str(label)
    if "\\boxed" in truth:
        truth = extract_answer(truth)
        if truth is None:
            return 0
    if grade_answer_mathd(model_answer, truth) or grade_answer_sympy(model_answer, truth):
        return 1
    return 0


async def sao_math_rm(args, sample, **kwargs) -> int:
    return grade_response(sample.response or "", sample.label)
