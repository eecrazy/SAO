"""Base-pass-rate filter campaign for paper_006 (experiment_spec C7, A16).

For each candidate prompt (data_prep.py output), draw n samples from
Qwen3.5-4B via a running sglang server (OpenAI-compatible /v1/chat/completions
— the SAME chat template the trainer applies), grade with
sao_plugin.reward_math.grade_response, and record the pass count. Then emit the
training pool: prompts with pass_count/n inside [--keep-min, --keep-max]
(default [1/8, 7/8] at n=8).

Resumable: results are appended one JSON line per finished problem to
--results (keyed by metadata.uid); reruns skip finished uids. Kill-safe under
the 4h partition cap — resubmit to continue (user preference: resumable
pipelines, cache keyed by stable ids).

Outputs (with --emit):
  <out>            train pool jsonl (same schema as candidates)
  <out>.stats.json histogram + yield summary

Usage (inside the slime container, after sglang server is up):
  python filter_campaign.py --candidates c.jsonl --results r.jsonl \
      --server http://127.0.0.1:30000 --n 8 --max-tokens 24576 [--emit pool.jsonl]
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sao_plugin.prompt import BOXED_INSTRUCTION  # noqa: E402

# grade_response pulls in slime (container-only). Import it lazily so `--emit`
# (pure aggregation over precomputed pass-counts) runs anywhere, e.g. the ais env.
grade_response = None

# Prompt-format tag stamped on every filter record. The candidate uid is derived
# from the BARE question (format-independent, for cross-run dedup), and the resume
# cache keys on uid — so WITHOUT this tag, regenerating candidates.jsonl with a new
# prompt format (e.g. adding the boxing instruction) would silently reuse stale
# pass-counts scored on the OLD format (fidelity review wf_a0d05795). We stamp the
# format each record was scored under and only treat a uid as "done" if its tag
# matches the current format, so a format change auto-invalidates the cache.
INSTRUCTED_TAG = "boxed-" + hashlib.sha1(BOXED_INSTRUCTION.encode()).hexdigest()[:8]


def _prompt_format_tag(prompt: str) -> str:
    return INSTRUCTED_TAG if BOXED_INSTRUCTION in (prompt or "") else "plain"


async def sample_once(session, server, model, prompt, max_tokens, temperature, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    async with session.post(
        f"{server}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    choice = data["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason") or ""


def _has_final_answer(text):
    """True if the response emitted a boxed answer in the post-</think> region
    (or anywhere, if the think block never closed) — i.e. produced *some* answer
    rather than running out of tokens mid-reasoning."""
    solution = text.split("</think>")[-1] if "</think>" in text else text
    return "\\boxed" in solution


async def process_problem(sem, session, args, row, results_f, lock):
    uid = row["metadata"]["uid"]
    passes, errors = 0, 0
    async with sem:
        tasks = [
            sample_once(session, args.server, args.model, row["prompt"], args.max_tokens, args.temperature, args.timeout)
            for _ in range(args.n)
        ]
        outs = await asyncio.gather(*tasks, return_exceptions=True)
    lengths = []
    truncated = 0  # finish_reason == 'length' → ran out of tokens
    has_answer = 0  # produced a \boxed answer (regardless of correctness)
    for out in outs:
        if isinstance(out, Exception):
            errors += 1
            continue
        content, finish_reason = out
        lengths.append(len(content))
        if finish_reason == "length":
            truncated += 1
        if _has_final_answer(content):
            has_answer += 1
        passes += grade_response(content, row["label"])
    record = {
        "uid": uid,
        "n": args.n,
        "errors": errors,
        "passes": passes,
        "truncated": truncated,
        "has_answer": has_answer,
        "mean_chars": sum(lengths) / len(lengths) if lengths else 0,
        "d32": (row.get("metadata") or {}).get("d32"),
        "pfmt": _prompt_format_tag(row.get("prompt", "")),
    }
    async with lock:
        results_f.write(json.dumps(record) + "\n")
        results_f.flush()
    return record


async def run_campaign(args, rows, done):
    todo = [r for r in rows if r["metadata"]["uid"] not in done]
    print(f"todo {len(todo)} / {len(rows)} (resumed {len(done)})")
    sem = asyncio.Semaphore(args.problem_concurrency)
    lock = asyncio.Lock()
    connector = aiohttp.TCPConnector(limit=args.problem_concurrency * args.n + 16)
    finished = 0
    with open(args.results, "a") as results_f:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [asyncio.create_task(process_problem(sem, session, args, row, results_f, lock)) for row in todo]
            try:
                for fut in asyncio.as_completed(tasks):
                    await fut
                    finished += 1
                    if finished % 100 == 0:
                        print(f"finished {finished}/{len(todo)}", flush=True)
            finally:
                # Keep the append file open until in-flight requests have been
                # cancelled and reaped. Otherwise Ctrl-C can let a task finish
                # after the context manager closes results_f.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--server", default=os.environ.get("SAO_FILTER_SERVER", "http://127.0.0.1:30000"))
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--problem-concurrency", type=int, default=48)
    ap.add_argument("--keep-min", type=float, default=1 / 8)
    ap.add_argument("--keep-max", type=float, default=7 / 8)
    ap.add_argument("--emit", default=None, help="write the filtered train pool jsonl and exit (no sampling)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.candidates)]
    # Current prompt-format tag per candidate uid — a cached result is only valid
    # if it was scored under the SAME format as the current candidate prompt.
    cand_fmt = {r["metadata"]["uid"]: _prompt_format_tag(r.get("prompt", "")) for r in rows}
    done = {}
    stale = 0
    if os.path.exists(args.results):
        for line in open(args.results):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A problem is final only if all n samples succeeded (error'd ones re-run)
            # AND it was scored under the current prompt format (else re-run — a
            # format-blind cache would silently reuse stale pass-counts, wf_a0d05795).
            if rec.get("errors", 0) == 0 and rec.get("n", 0) >= args.n:
                if rec.get("pfmt") == cand_fmt.get(rec["uid"]):
                    done[rec["uid"]] = rec
                else:
                    stale += 1
    if stale:
        print(f"WARNING: {stale} cached results ignored — scored under a different prompt format", flush=True)

    if args.emit:
        hist = Counter(rec["passes"] for rec in done.values())
        kept, missing, unformatted = 0, 0, 0
        emit_tmp = args.emit + ".tmp"
        stats_path = args.emit + ".stats.json"
        stats_tmp = stats_path + ".tmp"
        with open(emit_tmp, "w") as f:
            for row in rows:
                rec = done.get(row["metadata"]["uid"])
                if rec is None:
                    missing += 1
                    continue
                rate = rec["passes"] / rec["n"]
                if args.keep_min <= rate <= args.keep_max:
                    if BOXED_INSTRUCTION not in row.get("prompt", ""):
                        unformatted += 1  # pre-training gate: pool must match eval format
                        continue
                    row["metadata"]["base_pass"] = rec["passes"]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
        stats = {
            "candidates": len(rows),
            "graded": len(done),
            "ungraded": missing,
            "kept": kept,
            "band": [args.keep_min, args.keep_max],
            "pass_histogram": dict(sorted(hist.items())),
        }
        print(json.dumps(stats, indent=2))
        if unformatted:
            os.remove(emit_tmp)
            raise SystemExit(
                f"ABORT: {unformatted} in-band prompts lack the boxing instruction — "
                f"candidates.jsonl is stale (regenerate with data_prep.py); refusing to emit a "
                f"pool whose format diverges from eval."
            )
        if missing:
            os.remove(emit_tmp)
            raise SystemExit(
                f"INCOMPLETE: {missing} candidates are still ungraded; rerun the sampling stage "
                "before emitting the training pool."
            )
        with open(stats_tmp, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(emit_tmp, args.emit)
        os.replace(stats_tmp, stats_path)
        return

    global grade_response
    from sao_plugin.reward_math import grade_response  # container-only; needed for sampling
    asyncio.run(run_campaign(args, rows, done))


if __name__ == "__main__":
    main()
