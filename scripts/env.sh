#!/usr/bin/env bash
# Shared paths and single-node defaults for this 8 x NVIDIA H20 host.

set -o pipefail

SAO_ROOT="${SAO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export SAO_ROOT
if [[ -f "$SAO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SAO_ROOT/.env"
  set +a
fi
export SB="${SB:-$SAO_ROOT}"
export CODE="${CODE:-$SAO_ROOT}"
export VIRTUAL_ENV="${SAO_VIRTUAL_ENV:-$SAO_ROOT/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export SLIME_DIR="${SLIME_DIR:-$SAO_ROOT/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-$SAO_ROOT/Megatron-LM}"
export PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR:$SAO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export RUNS="${RUNS:-$SAO_ROOT/runs}"
export SAO_RUN_DIR_ROOT="${SAO_RUN_DIR_ROOT:-$RUNS}"
export HF_HOME="${HF_HOME:-$SAO_ROOT/.cache/huggingface}"
export HF_MODEL="${HF_MODEL:-$SAO_ROOT/artifacts/models/Qwen3.5-4B}"
export TORCH_DIST="${TORCH_DIST:-$SAO_ROOT/artifacts/models/Qwen3.5-4B_torch_dist}"
export DATA="${DATA:-$SAO_ROOT/data/pool.jsonl}"
export BENCH_DIR="${BENCH_DIR:-$SAO_ROOT/data/bench}"

# Local hardware: 8 H20 (96 GiB each), all-to-all NV18. Keep the validated
# single-node topology: actor+critic on four GPUs and four 1-GPU SGLang engines.
export NUM_GPUS="${NUM_GPUS:-8}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
export TP="${TP:-2}"
export CP="${CP:-1}"
export USE_DIST_OPT="${USE_DIST_OPT:-0}"
export SGLANG_CONCURRENCY="${SGLANG_CONCURRENCY:-24}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-24576}"
export SAO_MAX_WEIGHT_STALENESS="${SAO_MAX_WEIGHT_STALENESS:-4}"

SAO_BUNDLED_CUDA="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
if [[ -x "$SAO_BUNDLED_CUDA/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$SAO_BUNDLED_CUDA}"
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
fi
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

NVIDIA_SITE="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
if [[ -d "$NVIDIA_SITE/cu13" ]]; then
  export CPATH="$NVIDIA_SITE/cu13/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$NVIDIA_SITE/cu13/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$NVIDIA_SITE/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$NVIDIA_SITE/nccl" ]]; then
  export CPATH="$NVIDIA_SITE/nccl/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$NVIDIA_SITE/nccl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$NVIDIA_SITE/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$NVIDIA_SITE/cudnn" ]]; then
  export CPATH="$NVIDIA_SITE/cudnn/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$NVIDIA_SITE/cudnn/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$NVIDIA_SITE/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$RUNS" "$HF_HOME" "$SAO_ROOT/artifacts/models" "$SAO_ROOT/data" "$SAO_ROOT/logs"
