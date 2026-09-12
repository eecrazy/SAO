"""paper_006 eval driver — pass@1 (mean over n samples/problem) against an
OpenAI-compatible server (sglang serving an HF checkpoint).

Protocol (experiment_spec): temp 1.0, top_p 1.0, max 32k tokens, single turn,
no tools; grading = sao_plugin.reward_math.grade_response (same fn as training
reward — train==eval grading parity). Per-SAMPLE results are appended
incrementally to --results (uid, sample_idx, pass, chars) → fully resumable and
supports paired per-problem arm-vs-arm statistics later (L016).

  sample:    python run_eval.py sample --bench b.jsonl --results r.jsonl \
                 --server http://... --n 64 --max-tokens 32768
  aggregate: python run_eval.py aggregate --results r1.jsonl [r2.jsonl ...]
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def sample_once(session, args, prompt):
    import aiohttp

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    async with session.post(
        f"{args.server}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["choices"][0]["message"].get("content") or ""


async def run_sampling(args):
    import aiohttp
    from sao_plugin.reward_math import grade_response

    rows = [json.loads(line) for line in open(args.bench)]
    done = defaultdict(set)
    if os.path.exists(args.results):
        for line in open(args.results):
            try:
                rec = json.loads(line)
                done[rec["uid"]].add(rec["sample_idx"])
            except (json.JSONDecodeError, KeyError):
                continue

    todo = []
    for row in rows:
        uid = row["metadata"]["uid"]
        for i in range(args.n):
            if i not in done[uid]:
                todo.append((row, i))
    print(f"todo {len(todo)} samples over {len(rows)} problems (resumed)")

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    finished = 0

    async def one(row, idx, results_f, session):
        nonlocal finished
        async with sem:
            try:
                out = await sample_once(session, args, row["prompt"])
            except Exception as e:  # noqa: BLE001
                print(f"sample error uid={row['metadata']['uid']} idx={idx}: {type(e).__name__}", flush=True)
                return
        rec = {
            "uid": row["metadata"]["uid"],
            "bench": row["metadata"]["bench"],
            "sample_idx": idx,
            "pass": grade_response(out, row["label"]),
            "chars": len(out),
        }
        async with lock:
            results_f.write(json.dumps(rec) + "\n")
            results_f.flush()
            finished += 1
            if finished % 200 == 0:
                print(f"finished {finished}/{len(todo)}", flush=True)

    connector = aiohttp.TCPConnector(limit=args.concurrency + 16)
    with open(args.results, "a") as results_f:
        async with aiohttp.ClientSession(connector=connector) as session:
            await asyncio.gather(*[one(row, i, results_f, session) for row, i in todo])


def aggregate(paths, n_expected=None):
    per_problem = defaultdict(lambda: defaultdict(dict))  # bench -> uid -> idx -> pass
    for path in paths:
        for line in open(path):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            per_problem[rec["bench"]][rec["uid"]][rec["sample_idx"]] = rec["pass"]

    summary = {}
    for bench, problems in sorted(per_problem.items()):
        scores = []
        counts = []
        for uid, samples in problems.items():
            vals = list(samples.values())
            counts.append(len(vals))
            scores.append(sum(vals) / len(vals))
        summary[bench] = {
            "problems": len(scores),
            "min_samples": min(counts),
            "max_samples": max(counts),
            "pass1": 100.0 * sum(scores) / len(scores),
        }
    if all(b in summary for b in ("aime25", "beyondaime", "hmmt25")):
        summary["avg_pass1_3bench"] = round(
            (summary["aime25"]["pass1"] + summary["beyondaime"]["pass1"] + summary["hmmt25"]["pass1"]) / 3, 3
        )
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--bench", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--server", default=os.environ.get("SAO_EVAL_SERVER", "http://127.0.0.1:30000"))
    s.add_argument("--model", default="sao-eval")
    s.add_argument("--n", type=int, default=64)
    s.add_argument("--max-tokens", type=int, default=32768)
    s.add_argument("--timeout", type=float, default=5400)
    s.add_argument("--concurrency", type=int, default=256)

    a = sub.add_parser("aggregate")
    a.add_argument("--results", nargs="+", dest="results_list", required=True)

    args = ap.parse_args()
    if args.cmd == "sample":
        asyncio.run(run_sampling(args))
    else:
        aggregate(args.results_list)


if __name__ == "__main__":
    main()
