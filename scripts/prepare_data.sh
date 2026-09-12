#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PYTHON="$VIRTUAL_ENV/bin/python"

mkdir -p "$ROOT/data" "$BENCH_DIR"
"$PYTHON" "$ROOT/sao_plugin/data_prep.py" \
  --out "$ROOT/data/candidates.jsonl" \
  --stratified --per-tier "${PER_TIER:-2000}" \
  --d32-min "${D32_MIN:-0}" --d32-max "${D32_MAX:-4}" \
  --seed "${DATA_SEED:-42}"
"$PYTHON" "$ROOT/eval/prep_benchmarks.py" --out-dir "$BENCH_DIR"

test -s "$ROOT/data/candidates.jsonl"
for bench in aime25 beyondaime hmmt25 math500; do test -s "$BENCH_DIR/$bench.jsonl"; done
wc -l "$ROOT/data/candidates.jsonl" "$BENCH_DIR"/*.jsonl
echo "DATA_CANDIDATES_AND_BENCHMARKS_READY"
