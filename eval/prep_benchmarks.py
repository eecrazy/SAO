"""Materialize the paper_006 eval benchmarks as jsonl (prompt/label/metadata).

Benchmarks (experiment_spec target metric):
  aime25      opencompass/AIME2025 (30 problems)
  beyondaime  ByteDance-Seed/BeyondAIME (100 problems)
  hmmt25      CMU-AIRe/hmmt-aime-2025, rows with data_source containing 'hmmt'
              (HMMT Feb 2025 — logged deviation from the paper's Nov-2025 set)
  math500     HuggingFaceH4/MATH-500 (saturated harness anchor, L012)

Column names are auto-detected defensively (question/problem/prompt;
answer/label/solution). Output: <out_dir>/<bench>.jsonl with
{"prompt", "label", "metadata": {"uid", "bench"}}.

Usage (ais env): python prep_benchmarks.py --out-dir $RUNS/data/bench
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sao_plugin.prompt import with_instruction  # noqa: E402

QUESTION_KEYS = ["question", "problem", "prompt", "Question", "Problem"]
ANSWER_KEYS = ["answer", "label", "final_answer", "Answer", "expected_answer"]
DATASET_REVISIONS = {
    "opencompass/AIME2025": "a6ad95f611d72cf628a80b58bd0432ef6638f958",
    "ByteDance-Seed/BeyondAIME": "c705198ae1043810b1e1693bd879250b51a7a523",
    "CMU-AIRe/hmmt-aime-2025": "7bdf466a5cecd539966303cfefe01380a296819b",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
}


def pick(row, keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)):
            return str(v)
    return None


def emit(rows, path, bench):
    n = 0
    with open(path, "w") as f:
        for row in rows:
            q = pick(row, QUESTION_KEYS)
            a = pick(row, ANSWER_KEYS)
            if not q or a is None:
                continue
            uid = hashlib.sha1(f"{bench}|{q}".encode()).hexdigest()[:16]
            f.write(
                json.dumps(
                    {"prompt": with_instruction(q), "label": a, "metadata": {"uid": uid, "bench": bench}},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"{bench}: wrote {n} problems -> {path}")
    if n == 0:
        raise SystemExit(f"{bench}: 0 problems extracted — column auto-detect failed, inspect the dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from datasets import load_dataset

    def all_rows(path, *cfg):
        ds = load_dataset(path, *cfg, revision=DATASET_REVISIONS[path])
        for split in ds:
            yield from ds[split]

    aime25 = list(all_rows("opencompass/AIME2025", "AIME2025-I")) + list(
        all_rows("opencompass/AIME2025", "AIME2025-II")
    )
    emit(aime25, os.path.join(args.out_dir, "aime25.jsonl"), "aime25")
    emit(all_rows("ByteDance-Seed/BeyondAIME"), os.path.join(args.out_dir, "beyondaime.jsonl"), "beyondaime")
    hmmt = [
        r
        for r in all_rows("CMU-AIRe/hmmt-aime-2025")
        if "hmmt" in str(r.get("data_source", r.get("source", ""))).lower()
    ]
    emit(hmmt, os.path.join(args.out_dir, "hmmt25.jsonl"), "hmmt25")
    emit(all_rows("HuggingFaceH4/MATH-500"), os.path.join(args.out_dir, "math500.jsonl"), "math500")


if __name__ == "__main__":
    main()
