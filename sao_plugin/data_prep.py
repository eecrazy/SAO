"""Skywork-OR1 math → SAO candidate pool (paper_006, experiment_spec C7).

Selects a hard-band candidate set from Skywork/Skywork-OR1-RL-Data (math split,
105,055 rows; verl format: prompt=[{role:user,content}], reward_model.ground_truth
= JSON list string, extra_info.model_difficulty = fails/16 for R1-Distill
1.5B/7B/32B) for the base-pass-rate filter campaign (filter_campaign.py — the
real difficulty arbiter is Qwen3.5-4B itself; this pre-band only cuts rollout
cost, INT-001/A16).

Keeps single-ground-truth rows only (multi-answer dropped, counted), dedupes by
normalized question, restricts to a difficulty band on the R1-Distill-32B label
(defaults: fails in [2,14] — excludes trivial and likely-broken/impossible),
sorts hardest-first (d32, then d7) with a seeded shuffle for ties, truncates to
--max-candidates, and writes jsonl rows:

    {"prompt": <question str>, "label": <answer str>,
     "metadata": {"uid","source","d15","d7","d32"}}

Consumed by slime with --input-key prompt --label-key label
--apply-chat-template (same template applied by the eval harness → parity).

Usage (ais env):
    python data_prep.py --out candidates.jsonl [--max-candidates 12000]
                        [--d32-min 2 --d32-max 14] [--seed 42]
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sao_plugin.prompt import with_instruction  # noqa: E402

SKYWORK_DATASET = "Skywork/Skywork-OR1-RL-Data"
SKYWORK_REVISION = "1cdedc52e0e2db85fdf252f9be682e63a5a38c33"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-candidates", type=int, default=12000)
    ap.add_argument("--d32-min", type=int, default=2)
    ap.add_argument("--d32-max", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    # Stratified mode: instead of hardest-first truncation (which collapses the
    # pool onto the hardest tier), take an equal --per-tier sample from each d32
    # value in [d32-min, d32-max]. Used for the calibration probe (locate the
    # 4B's learnable band) and for re-selecting a difficulty-balanced pool.
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--per-tier", type=int, default=48)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(SKYWORK_DATASET, revision=SKYWORK_REVISION, split="math")
    print(f"loaded math split: {ds.num_rows} rows")

    stats = Counter()
    seen: set[str] = set()
    rows = []
    for r in ds:
        stats["total"] += 1
        try:
            question = r["prompt"][0]["content"].strip()
        except (KeyError, IndexError, TypeError):
            stats["bad_prompt"] += 1
            continue
        try:
            truths = json.loads(r["reward_model"]["ground_truth"])
        except (json.JSONDecodeError, KeyError, TypeError):
            stats["bad_label"] += 1
            continue
        if not isinstance(truths, list) or len(truths) != 1 or not str(truths[0]).strip():
            stats["multi_or_empty_label"] += 1
            continue
        label = str(truths[0]).strip()

        key = hashlib.sha1(" ".join(question.split()).lower().encode()).hexdigest()
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)

        md = (r.get("extra_info") or {}).get("model_difficulty") or {}
        d15 = md.get("DeepSeek-R1-Distill-Qwen-1.5B")
        d7 = md.get("DeepSeek-R1-Distill-Qwen-7B")
        d32 = md.get("DeepSeek-R1-Distill-Qwen-32B")
        if d32 is None:
            stats["no_difficulty"] += 1
            continue
        if not (args.d32_min <= d32 <= args.d32_max):
            stats["out_of_band"] += 1
            continue

        stats["kept"] += 1
        rows.append(
            {
                "prompt": with_instruction(question),
                "label": label,
                "metadata": {
                    "uid": key[:16],
                    "source": r.get("data_source", ""),
                    "d15": d15,
                    "d7": d7,
                    "d32": d32,
                },
            }
        )

    print(dict(stats))
    hist = Counter(row["metadata"]["d32"] for row in rows)
    print("d32 histogram of kept:", dict(sorted(hist.items())))

    rng = random.Random(args.seed)
    rng.shuffle(rows)  # deterministic tie-break / per-tier sampling order
    if args.stratified:
        by_tier: dict[int, list] = {}
        for row in rows:
            by_tier.setdefault(row["metadata"]["d32"], []).append(row)
        picked = []
        for d in sorted(by_tier):
            picked.extend(by_tier[d][: args.per_tier])
        rows = picked
        got = Counter(row["metadata"]["d32"] for row in rows)
        print(f"stratified: {args.per_tier}/tier over d32 in [{args.d32_min},{args.d32_max}] -> {dict(sorted(got.items()))}")
    else:
        rows.sort(key=lambda row: (row["metadata"]["d32"], row["metadata"]["d7"] or 0), reverse=True)
        rows = rows[: args.max_candidates]
    print(f"writing {len(rows)} candidates ({'stratified' if args.stratified else 'hardest-first'}) -> {args.out}")

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
