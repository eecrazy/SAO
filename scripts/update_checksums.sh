#!/usr/bin/env bash
# Regenerate the integrity manifest for committed reproduction inputs/results.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p reproducibility
tmp="$(mktemp reproducibility/SHA256SUMS.XXXXXX)"
trap 'rm -f "$tmp"' EXIT

files=(
  data/candidates.jsonl
  data/filter_results.jsonl
  data/pool.jsonl
  data/pool.jsonl.stats.json
  data/bench/aime25.jsonl
  data/bench/beyondaime.jsonl
  data/bench/hmmt25.jsonl
  data/bench/math500.jsonl
  runs/eval/base/aime25.jsonl
  runs/eval/base/beyondaime.jsonl
  runs/eval/base/hmmt25.jsonl
  runs/eval/base/math500.jsonl
  runs/eval/final_sao/aime25.jsonl
  runs/eval/final_sao/beyondaime.jsonl
  runs/eval/final_sao/hmmt25.jsonl
  runs/eval/final_sao/math500.jsonl
  runs/eval/final_grpo/aime25.jsonl
  runs/eval/final_grpo/beyondaime.jsonl
  runs/eval/final_grpo/hmmt25.jsonl
  runs/eval/final_grpo/math500.jsonl
  patches/slime_sao.patch
  requirements/runtime-cu130.in
  requirements/runtime-cu130.overrides.txt
  requirements/runtime-cu130.lock.txt
  requirements/native-cu130.txt
  sao_plugin/configs/common.sh
  sao_plugin/configs/sao.sh
  sao_plugin/configs/grpo_dis.sh
  scripts/setup_uv.sh
  reproducibility/datasets.json
  reproducibility/hardware.json
  reproducibility/h20_config.json
  reproducibility/h20_results.json
)

sha256sum "${files[@]}" > "$tmp"
mv "$tmp" reproducibility/SHA256SUMS
trap - EXIT
echo "wrote reproducibility/SHA256SUMS (${#files[@]} files)"
