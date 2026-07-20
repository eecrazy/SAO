#!/bin/bash
# paper_006_sao — submit N singleton-chained 4h training segments.
# Usage: chain.sh <ARM_CONFIG.sh> <N_SEGMENTS> [JOB_NAME]
#   e.g. chain.sh .../sao_plugin/configs/sao.sh 8 sao-tr-sao
# Abort a chain: scancel --name=<JOB_NAME>
# ⚠ Always set RUN_TAG when forking a run — a bare re-launch auto-resumes (and
#   thereby extends/clobbers) the existing ARM run dir (paper_003 footgun).
set -euo pipefail

ARM_CONFIG=${1:?arm config}
N=${2:?number of segments}
JOB_NAME=${3:-sao-tr-$(basename "$ARM_CONFIG" .sh)${RUN_TAG:+-${RUN_TAG}}}
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

for _ in $(seq "$N"); do
   sbatch --job-name="$JOB_NAME" --dependency=singleton \
      --export=ALL,ARM_CONFIG="$ARM_CONFIG" \
      "$HERE/submit_train.sbatch"
done
squeue -u "$USER" -n "$JOB_NAME" -o "%.10i %.20j %.8T %.10M %R"
