#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
MODEL_DIR="${1:?usage: eval_local.sh HF_MODEL_DIR TAG}"
TAG="${2:?usage: eval_local.sh HF_MODEL_DIR TAG}"
PYTHON="$VIRTUAL_ENV/bin/python"
PORT="${PORT:-30000}"
SERVER="http://127.0.0.1:$PORT"
OUT_DIR="$RUNS/eval/$TAG"
LOG="$ROOT/logs/eval_${TAG}_sglang.log"
mkdir -p "$OUT_DIR"
test -s "$MODEL_DIR/config.json"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name sao-eval \
  --dp-size 8 --tp-size 1 --mem-fraction-static 0.85 \
  --host 127.0.0.1 --port "$PORT" --log-level warning >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do
  curl -sf "$SERVER/health" >/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -200 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "$SERVER/health" >/dev/null || { tail -200 "$LOG"; exit 1; }

for bench in ${BENCHES:-aime25 beyondaime hmmt25 math500}; do
  case "$bench" in
    aime25) N="${N_AIME:-64}" ;;
    beyondaime) N="${N_BEYOND:-32}" ;;
    hmmt25) N="${N_HMMT:-64}" ;;
    math500) N="${N_MATH500:-8}" ;;
  esac
  "$PYTHON" "$ROOT/eval/run_eval.py" sample \
    --bench "$BENCH_DIR/$bench.jsonl" --results "$OUT_DIR/$bench.jsonl" \
    --server "$SERVER" --model sao-eval --n "$N" \
    --max-tokens "${EVAL_MAX_TOKENS:-32768}" --concurrency "${EVAL_CONCURRENCY:-256}"
done
"$PYTHON" "$ROOT/eval/run_eval.py" aggregate --results "$OUT_DIR"/*.jsonl
