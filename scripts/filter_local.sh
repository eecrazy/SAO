#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PYTHON="$VIRTUAL_ENV/bin/python"
CANDIDATES="${CANDIDATES:-$ROOT/data/candidates.jsonl}"
RESULTS="${RESULTS:-$ROOT/data/filter_results.jsonl}"
PORT="${PORT:-30000}"
SERVER="http://127.0.0.1:$PORT"
LOG="$ROOT/logs/filter_sglang.log"

test -s "$HF_MODEL/config.json"
test -s "$CANDIDATES"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid "$PYTHON" -m sglang.launch_server \
  --model-path "$HF_MODEL" --served-model-name qwen3.5-4b \
  --dp-size 8 --tp-size 1 --mem-fraction-static 0.85 \
  --host 127.0.0.1 --port "$PORT" --log-level warning >"$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  curl -sf "$SERVER/health" >/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -200 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "$SERVER/health" >/dev/null || { tail -200 "$LOG"; exit 1; }

"$PYTHON" "$ROOT/sao_plugin/filter_campaign.py" \
  --candidates "$CANDIDATES" --results "$RESULTS" \
  --server "$SERVER" --model qwen3.5-4b \
  --n "${FILTER_N:-8}" --max-tokens "${FILTER_MAX_TOKENS:-24576}" \
  --problem-concurrency "${FILTER_CONCURRENCY:-48}"

"$PYTHON" "$ROOT/sao_plugin/filter_campaign.py" \
  --candidates "$CANDIDATES" --results "$RESULTS" --emit "$DATA"
test -s "$DATA"
wc -l "$CANDIDATES" "$RESULTS" "$DATA"
echo "FILTERED_POOL_READY=$DATA"
