#!/usr/bin/env python3
"""Verify committed snapshot integrity, coverage, and reported H20 scores."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_SAMPLES = {"aime25": (30, 64), "beyondaime": (100, 32), "hmmt25": (30, 64), "math500": (500, 8)}


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{number}: invalid JSON") from exc


def verify_hashes() -> None:
    manifest = ROOT / "reproducibility/SHA256SUMS"
    for line in manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, f"SHA256 mismatch: {relative}"


def verify_data() -> None:
    candidates = list(jsonl(ROOT / "data/candidates.jsonl"))
    pool = list(jsonl(ROOT / "data/pool.jsonl"))
    candidate_uids = [row["metadata"]["uid"] for row in candidates]
    pool_uids = [row["metadata"]["uid"] for row in pool]
    assert len(candidates) == len(set(candidate_uids)) == 10_000
    assert len(pool) == len(set(pool_uids)) == 3_993
    assert set(pool_uids) <= set(candidate_uids)

    filters = list(jsonl(ROOT / "data/filter_results.jsonl"))
    latest = {row["uid"]: row for row in filters if row.get("errors") == 0 and row.get("n", 0) >= 8}
    assert len(filters) == 10_049
    assert set(latest) == set(candidate_uids)
    assert all(row.get("pfmt") == "boxed-a847bc0f" for row in latest.values())

    stats = json.loads((ROOT / "data/pool.jsonl.stats.json").read_text())
    assert (stats["candidates"], stats["graded"], stats["ungraded"], stats["kept"]) == (10_000, 10_000, 0, 3_993)
    assert stats["pass_histogram"] == {str(i): sum(row["passes"] == i for row in latest.values()) for i in range(9)}

    for bench, (problems, _) in BENCH_SAMPLES.items():
        rows = list(jsonl(ROOT / f"data/bench/{bench}.jsonl"))
        assert len(rows) == problems
        assert len({row["metadata"]["uid"] for row in rows}) == problems


def summarize_eval(arm: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    for bench, (problem_count, sample_count) in BENCH_SAMPLES.items():
        records = list(jsonl(ROOT / f"runs/eval/{arm}/{bench}.jsonl"))
        per_problem: dict[str, dict[int, int]] = defaultdict(dict)
        for row in records:
            per_problem[row["uid"]][row["sample_idx"]] = row["pass"]
        assert len(records) == problem_count * sample_count
        assert len(per_problem) == problem_count
        assert all(set(samples) == set(range(sample_count)) for samples in per_problem.values())
        summary[bench] = 100 * sum(sum(samples.values()) / sample_count for samples in per_problem.values()) / problem_count
    summary["avg_pass1_3bench"] = sum(summary[name] for name in ("aime25", "beyondaime", "hmmt25")) / 3
    return summary


def verify_results() -> None:
    recorded = json.loads((ROOT / "reproducibility/h20_results.json").read_text())["arms"]
    for arm in ("base", "sao", "grpo_dis"):
        actual = summarize_eval("final_grpo" if arm == "grpo_dis" else ("final_sao" if arm == "sao" else arm))
        for metric, value in actual.items():
            assert abs(value - recorded[arm][metric]) < 0.0005, f"result mismatch: {arm}/{metric}"
        print(f"{arm:8s} avg_pass1_3bench={actual['avg_pass1_3bench']:.3f}")


def main() -> None:
    verify_hashes()
    verify_data()
    verify_results()
    print("reproducibility snapshots: OK")


if __name__ == "__main__":
    main()
