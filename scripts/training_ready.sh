#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PYTHON="$VIRTUAL_ENV/bin/python"

"$ROOT/scripts/preflight.sh"

for path in \
  "$HF_MODEL/config.json" \
  "$TORCH_DIST/latest_checkpointed_iteration.txt" \
  "$ROOT/data/candidates.jsonl" \
  "$ROOT/data/filter_results.jsonl" \
  "$DATA" \
  "$DATA.stats.json" \
  "$RUNS/sao_smoke/ckpt/latest_checkpointed_iteration.txt" \
  "$RUNS/sao_smoke/critic_ckpt/latest_checkpointed_iteration.txt" \
  "$RUNS/grpo_dis_smoke/ckpt/latest_checkpointed_iteration.txt"; do
  test -s "$path" || { echo "training asset missing or empty: $path" >&2; exit 1; }
done

"$PYTHON" - "$ROOT/data/candidates.jsonl" "$ROOT/data/filter_results.jsonl" "$DATA" "$DATA.stats.json" <<'PY'
from collections import Counter
import json
import sys

from sao_plugin.filter_campaign import _prompt_format_tag
from sao_plugin.prompt import BOXED_INSTRUCTION

candidates_path, results_path, pool_path, stats_path = sys.argv[1:]
candidates = [json.loads(line) for line in open(candidates_path)]
pool = [json.loads(line) for line in open(pool_path)]
stats = json.load(open(stats_path))

candidate_by_uid = {row["metadata"]["uid"]: row for row in candidates}
assert len(candidate_by_uid) == len(candidates), "duplicate UIDs in candidates"
done = {}
for line in open(results_path):
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    uid = record.get("uid")
    candidate = candidate_by_uid.get(uid)
    if candidate is None:
        continue
    if (
        record.get("errors", 0) == 0
        and record.get("n") == 8
        and record.get("pfmt") == _prompt_format_tag(candidate.get("prompt", ""))
    ):
        done[uid] = record

assert stats["candidates"] == len(candidates), stats
assert stats["graded"] == len(done) == len(candidates), stats
assert stats["ungraded"] == 0, stats
assert stats["kept"] == len(pool) > 0, stats
expected_histogram = {str(key): value for key, value in sorted(Counter(r["passes"] for r in done.values()).items())}
assert stats["pass_histogram"] == expected_histogram, (stats["pass_histogram"], expected_histogram)

uids = [row["metadata"]["uid"] for row in pool]
assert len(uids) == len(set(uids)), "duplicate UIDs in filtered pool"
assert all(BOXED_INSTRUCTION in row.get("prompt", "") for row in pool), "uninstructed prompt in filtered pool"
assert all(1 <= int(row["metadata"]["base_pass"]) <= 7 for row in pool), "out-of-band base_pass"
expected_uids = {uid for uid, record in done.items() if 1 <= record["passes"] <= 7}
assert set(uids) == expected_uids, "pool membership differs from recomputed difficulty band"
for row in pool:
    uid = row["metadata"]["uid"]
    assert row["metadata"]["base_pass"] == done[uid]["passes"], f"base_pass mismatch for {uid}"
    assert row["prompt"] == candidate_by_uid[uid]["prompt"], f"prompt mismatch for {uid}"
    assert row["label"] == candidate_by_uid[uid]["label"], f"label mismatch for {uid}"
print(f"filtered pool: {len(pool)}/{len(candidates)} prompts, complete and format-checked")
PY

for latest in \
  "$RUNS/sao_smoke/ckpt/latest_checkpointed_iteration.txt" \
  "$RUNS/sao_smoke/critic_ckpt/latest_checkpointed_iteration.txt" \
  "$RUNS/grpo_dis_smoke/ckpt/latest_checkpointed_iteration.txt"; do
  iteration="$(<"$latest")"
  [[ "$iteration" =~ ^[0-9]+$ ]] && (( iteration >= 2 )) || {
    echo "smoke checkpoint did not reach iteration 2: $latest=$iteration" >&2
    exit 1
  }
done

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
MIN_GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)"
[[ "$GPU_COUNT" -eq "$NUM_GPUS" ]] || { echo "GPU count changed: $GPU_COUNT != $NUM_GPUS" >&2; exit 1; }
(( MIN_GPU_MIB >= 90000 )) || { echo "this profile requires at least 90,000 MiB per GPU" >&2; exit 1; }
(( ACTOR_GPUS + ROLLOUT_GPUS == NUM_GPUS )) || { echo "actor/rollout GPU split is inconsistent" >&2; exit 1; }
(( ACTOR_GPUS % TP == 0 )) || { echo "actor GPU count must be divisible by TP" >&2; exit 1; }

TMP_CONFIG_ROOT="$(mktemp -d)"
cleanup() { rm -rf -- "$TMP_CONFIG_ROOT"; }
trap cleanup EXIT
for arm in sao grpo_dis; do
  (
    export RUN_TAG=readiness
    export SAO_RUN_DIR_ROOT="$TMP_CONFIG_ROOT"
    unset NUM_ROLLOUT NUM_CRITIC_ONLY_STEPS MAX_TOKENS_PER_GPU
    # shellcheck disable=SC1090
    source "$ROOT/sao_plugin/configs/$arm.sh"
    common=" ${ROLLOUT_COMMON[*]} "
    algo=" ${ALGO_ARGS[*]} "
    ckpt=" ${CKPT_ARGS[*]} "
    perf=" ${PERF_ARGS[*]} "
    sglang=" ${SGLANG_ARGS[*]} "
    [[ "$ckpt" == *" --save-interval 10 "* ]]
    [[ "$common" == *" --global-batch-size 128 "* ]]
    [[ "$common" == *" --rollout-max-response-len 24576 "* ]]
    [[ "$common" == *" --update-weights-interval 1 "* ]]
    [[ " ${DIS_ARGS[*]} " == *" --eps-clip 0.3 "* ]]
    [[ " ${DIS_ARGS[*]} " == *" --eps-clip-high 5.0 "* ]]
    [[ "$perf" == *" --tensor-model-parallel-size 2 "* ]]
    [[ "$perf" == *" --context-parallel-size 1 "* ]]
    [[ "$perf" != *" --use-distributed-optimizer "* ]]
    [[ "$sglang" == *" --sglang-mem-fraction-static 0.85 "* ]]
    [[ "$sglang" == *" --sglang-server-concurrency 24 "* ]]
    [[ "$ACTOR_GPUS" -eq 4 && "$ROLLOUT_GPUS" -eq 4 ]]
    [[ "$SAO_MAX_WEIGHT_STALENESS" -eq 4 ]]
    if [[ "$arm" == sao ]]; then
      [[ "$NUM_ROLLOUT" -eq 547 ]]
      [[ "$algo" == *" --rollout-batch-size 128 "* ]]
      [[ "$algo" == *" --n-samples-per-prompt 1 "* ]]
      [[ "$algo" == *" --num-critic-only-steps 47 "* ]]
      [[ "$algo" == *" --custom-advantage-function-path sao_plugin.adaptive_gae.sao_adaptive_gae "* ]]
      [[ "$SAO_CRITIC_UPDATES_PER_STEP" -eq 2 ]]
      [[ "$perf" == *" --max-tokens-per-gpu 12288 "* ]]
      rg -q 'lr: 5.0e-6' "$ROLES_YAML"
      rg -q 'freeze_params_name_list: \["self_attention"\]' "$ROLES_YAML"
    else
      [[ "$NUM_ROLLOUT" -eq 500 ]]
      [[ "$algo" == *" --rollout-batch-size 16 "* ]]
      [[ "$algo" == *" --n-samples-per-prompt 8 "* ]]
      [[ "$algo" == *" --disable-grpo-std-normalization "* ]]
      [[ "$perf" == *" --max-tokens-per-gpu 32768 "* ]]
    fi
  )
done

MIN_TRAIN_FREE_TIB="${MIN_TRAIN_FREE_TIB:-10}"
FREE_BYTES="$(df --output=avail -B1 "$RUNS" | tail -1 | tr -d ' ')"
REQUIRED_BYTES=$(( MIN_TRAIN_FREE_TIB * 1024 * 1024 * 1024 * 1024 ))
(( FREE_BYTES >= REQUIRED_BYTES )) || {
  echo "need at least ${MIN_TRAIN_FREE_TIB} TiB free under $RUNS for both checkpoint series" >&2
  exit 1
}

echo "SAO_FULL_TRAINING_READY"
