#!/usr/bin/env bash
# Build the native SAO/slime CUDA environment with uv. This mirrors the
# slimerl/slime:nightly-dev-20260707a Dockerfile while remaining host-native.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

UV_CACHE_DIR="${UV_CACHE_DIR:-/data/github/.uv-cache}"
export UV_CACHE_DIR
PYTHON="$ROOT/.venv/bin/python"
MAX_JOBS="${MAX_JOBS:-32}"
export MAX_JOBS

SLIME_COMMIT=680824dd5e01a2e83750bf87fc366ec6fa98766c
MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
APEX_COMMIT=10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
TMS_COMMIT=a193d9dd1b877d33c64a41cfb3db9f867df2d926
MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
MEGATRON_BRIDGE_COMMIT=7f0fb3456f8ffe47599b5fd167b454605d85f932
FLASHQLA_COMMIT=afe07c406a5c9475ec66571ffd6c15c0358a947f
FA3_COMMIT=002cce0a1068f8c07dfccb5a1d232b9a3276947c
TE_COMMIT=c9877beb87ad7e711e1869dd0b5062167ede447a
TE_CUDNN_FRONTEND_COMMIT=97f6cb3b88cacff507cca1280db5650a457d92b3
TE_CUTLASS_COMMIT=57e3cfb47a2d9e0d46eb6335c3dc411498efa198
GITHUB_ARCHIVE_PROXY="${GITHUB_ARCHIVE_PROXY:-https://ghfast.top/https://github.com}"
GITHUB_ARCHIVE_DIRECT="${GITHUB_ARCHIVE_DIRECT:-https://github.com}"
GITHUB_RELEASE_PROXY="${GITHUB_RELEASE_PROXY:-https://ghfast.top/https://github.com}"

download_archive() {
  local repo="$1" revision="$2" output="$3"
  local proxy_url="$GITHUB_ARCHIVE_PROXY/$repo/archive/$revision.tar.gz"
  local direct_url="$GITHUB_ARCHIVE_DIRECT/$repo/archive/$revision.tar.gz"
  local curl_args=(-fL --retry 2 --connect-timeout 15 --speed-limit 1024 --speed-time 30)
  if ! curl "${curl_args[@]}" "$proxy_url" -o "$output"; then
    [[ "$proxy_url" != "$direct_url" ]] || return 1
    echo "Archive proxy failed; retrying directly from GitHub: $repo@$revision" >&2
    curl "${curl_args[@]}" "$direct_url" -o "$output"
  fi
}

fetch_archive() {
  local repo="$1" revision="$2" destination="$3"
  local marker="$destination/.source-revision"
  if [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$revision" ]]; then
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "Refusing to replace $destination: remove or relocate it, then rerun." >&2
    exit 1
  fi
  local archive="$ROOT/.cache/sources/${repo//\//--}-$revision.tar.gz"
  local staging="$destination.staging.$revision"
  mkdir -p "$(dirname "$archive")" "$staging"
  download_archive "$repo" "$revision" "$archive"
  tar -xzf "$archive" --strip-components=1 -C "$staging"
  printf '%s\n' "$revision" > "$staging/.source-revision"
  mv "$staging" "$destination"
}

verify_source_revision() {
  local destination="$1" expected="$2" actual=""
  if [[ -f "$destination/.source-revision" ]]; then
    actual="$(<"$destination/.source-revision")"
  elif git -C "$destination" rev-parse --verify HEAD >/dev/null 2>&1; then
    actual="$(git -C "$destination" rev-parse HEAD)"
  fi
  [[ "$actual" == "$expected" ]] || {
    echo "Source revision mismatch for $destination: ${actual:-unknown} != $expected" >&2
    echo "Remove or relocate that directory, then rerun setup." >&2
    exit 1
  }
}

fetch_submodule_archive() {
  local repo="$1" revision="$2" destination="$3"
  local marker="$destination/.source-revision"
  if [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$revision" ]]; then
    return
  fi
  mkdir -p "$destination"
  if [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to replace populated submodule directory $destination" >&2
    exit 1
  fi
  local archive="$ROOT/.cache/sources/${repo//\//--}-$revision.tar.gz"
  mkdir -p "$(dirname "$archive")"
  download_archive "$repo" "$revision" "$archive"
  tar -xzf "$archive" --strip-components=1 -C "$destination"
  printf '%s\n' "$revision" > "$marker"
}

apply_patch_once() {
  local directory="$1" strip="$2" patch_file="$3"
  if patch --batch --forward --dry-run -d "$directory" -p"$strip" < "$patch_file" >/dev/null 2>&1; then
    patch --batch --forward -d "$directory" -p"$strip" < "$patch_file"
  elif patch --batch --reverse --dry-run -d "$directory" -p"$strip" < "$patch_file" >/dev/null 2>&1; then
    echo "already applied: $patch_file"
  else
    echo "Patch is neither applicable nor already applied: $patch_file" >&2
    exit 1
  fi
}

sglang_patch_stack_is_applied() {
  local site="$1"
  shift
  local patches=("$@")
  local staging stack_status=0 i
  staging="$(mktemp -d)"
  cp -a "$site/sglang" "$staging/sglang"
  for ((i=${#patches[@]} - 1; i >= 0; i--)); do
    if ! patch --batch --reverse -d "$staging" -p2 < "${patches[$i]}" >/dev/null 2>&1; then
      stack_status=1
      break
    fi
  done
  find "$staging" -depth -delete
  return "$stack_status"
}

command -v uv >/dev/null
command -v nvidia-smi >/dev/null
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "nvcc not found under CUDA_HOME=$CUDA_HOME" >&2; exit 1; }
HOST_CUDA_HOME="$CUDA_HOME"
[[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u)" == "9.0" ]] || {
  echo "This profile expects homogeneous SM90 GPUs; override TORCH_CUDA_ARCH_LIST if intentional." >&2
  exit 1
}

if [[ ! -x "$PYTHON" ]]; then
  uv venv --python 3.12 "$ROOT/.venv"
fi
uv sync --frozen --inexact 2>/dev/null || uv sync --inexact

# The reference image uses cu129. This host has a CUDA 13.3 toolkit and a 580
# driver, while SGLang 0.5.13's published wheel natively targets torch 2.11/cu130.
# Use that supported CUDA-13 path to avoid mixing a cu129 runtime with nvcc 13.3.
# Install the complete registry resolution before source builds so a later
# dependency install cannot silently replace a pinned native component.
uv pip install --python "$PYTHON" \
  --requirements "$ROOT/requirements/runtime-cu130.lock.txt" \
  --require-hashes --no-deps --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.nvidia.com \
  --find-links https://tile-ai.github.io/whl/nightly/cu128/
uv pip uninstall --python "$PYTHON" flash-attn-4 flash_attn_4 || true

# From this point compile against the CUDA prefix shipped with the cu130 wheels,
# not the host's newer CUDA 13.3 libraries. This prevents same-SONAME ABI skew.
export CUDA_HOME="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ -e "$CUDA_HOME/lib64" ]] || ln -s lib "$CUDA_HOME/lib64"
[[ -e "$CUDA_HOME/lib/libcudart.so" ]] || ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
[[ -e "$CUDA_HOME/lib/libcublas.so" ]] || ln -s libcublas.so.13 "$CUDA_HOME/lib/libcublas.so"
[[ -e "$CUDA_HOME/lib/libcublasLt.so" ]] || ln -s libcublasLt.so.13 "$CUDA_HOME/lib/libcublasLt.so"
# CUDA 13 runtime wheels omit this deprecated compatibility header, although
# TE 2.16 still includes it (without using its APIs). Source the matching shim
# from the host toolkit when assembling the wheel-based CUDA prefix.
if [[ ! -e "$CUDA_HOME/include/cuda_profiler_api.h" ]]; then
  for profiler_header in \
    "$HOST_CUDA_HOME/targets/x86_64-linux/include/cuda_profiler_api.h" \
    "$HOST_CUDA_HOME/include/cuda_profiler_api.h"; do
    if [[ -f "$profiler_header" ]]; then
      ln -s "$profiler_header" "$CUDA_HOME/include/cuda_profiler_api.h"
      break
    fi
  done
fi
[[ -e "$CUDA_HOME/include/cuda_profiler_api.h" ]] || {
  echo "cuda_profiler_api.h not found in bundled or host CUDA toolkit" >&2
  exit 1
}
if [[ ! -e "$CUDA_HOME/include/nvml.h" ]]; then
  for nvml_header in \
    "$HOST_CUDA_HOME/targets/x86_64-linux/include/nvml.h" \
    "$HOST_CUDA_HOME/include/nvml.h"; do
    if [[ -f "$nvml_header" ]]; then
      ln -s "$nvml_header" "$CUDA_HOME/include/nvml.h"
      break
    fi
  done
fi
[[ -e "$CUDA_HOME/include/nvml.h" ]] || {
  echo "nvml.h not found in bundled or host CUDA toolkit" >&2
  exit 1
}

# Native kernels used by Qwen3.5 and Megatron.
uv pip install --python "$PYTHON" --no-deps --no-build-isolation \
  --require-hashes --requirements "$ROOT/requirements/native-cu130.txt"
fetch_archive NVIDIA/TransformerEngine "$TE_COMMIT" "$ROOT/third_party/TransformerEngine"
fetch_submodule_archive NVIDIA/cudnn-frontend "$TE_CUDNN_FRONTEND_COMMIT" \
  "$ROOT/third_party/TransformerEngine/3rdparty/cudnn-frontend"
fetch_submodule_archive NVIDIA/cutlass "$TE_CUTLASS_COMMIT" \
  "$ROOT/third_party/TransformerEngine/3rdparty/cutlass"
if "$PYTHON" - <<'PY'
from importlib.metadata import version

import torch
import transformer_engine.pytorch as te

assert version("transformer-engine").startswith("2.16.1")
x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
layer = te.Linear(16, 16, params_dtype=torch.bfloat16, device="cuda")
assert torch.isfinite(layer(x)).all()
PY
then
  echo "Transformer Engine 2.16.1 CUDA smoke test passed; skipping rebuild."
else
  NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=90 \
    CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" uv pip install --python "$PYTHON" \
    --reinstall --no-deps --no-build-isolation "$ROOT/third_party/TransformerEngine"
fi

fetch_archive ISEEKYAN/mbridge "$MBRIDGE_COMMIT" "$ROOT/third_party/mbridge"
uv pip install --python "$PYTHON" --no-deps "$ROOT/third_party/mbridge"
fetch_archive QwenLM/FlashQLA "$FLASHQLA_COMMIT" "$ROOT/third_party/FlashQLA"
uv pip install --python "$PYTHON" --no-deps --no-build-isolation "$ROOT/third_party/FlashQLA"

fetch_archive NVIDIA/apex "$APEX_COMMIT" "$ROOT/third_party/apex"
if "$PYTHON" -c 'import apex, amp_C, fused_layer_norm_cuda' >/dev/null 2>&1; then
  echo "Apex CUDA extensions import successfully; skipping rebuild."
else
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=8 NVCC_APPEND_FLAGS="--threads 4" \
    uv pip install --python "$PYTHON" --no-deps --no-build-isolation --no-cache "$ROOT/third_party/apex"
fi

# Fetch slime before Megatron: the pinned slime tree supplies the Megatron and
# SGLang patches consumed below.  Keeping this ahead of the first patch read is
# required for setup to work from a clean clone.
if [[ -d "$SLIME_DIR" ]]; then
  verify_source_revision "$SLIME_DIR" "$SLIME_COMMIT"
else
  fetch_archive THUDM/slime "$SLIME_COMMIT" "$SLIME_DIR"
fi

fetch_archive NVIDIA/Megatron-LM "$MEGATRON_COMMIT" "$MEGATRON_DIR"
apply_patch_once "$MEGATRON_DIR" 1 "$SLIME_DIR/docker/patch/latest/megatron.patch"
uv pip install --python "$PYTHON" --no-deps --no-build-isolation --editable "$MEGATRON_DIR"

TMS_CUDA_MAJOR=13
export TMS_CUDA_MAJOR
fetch_archive fzyzcjy/torch_memory_saver "$TMS_COMMIT" "$ROOT/third_party/torch_memory_saver"
uv pip install --python "$PYTHON" --reinstall --no-deps --no-cache --no-build-isolation "$ROOT/third_party/torch_memory_saver"
fetch_archive radixark/Megatron-Bridge "$MEGATRON_BRIDGE_COMMIT" "$ROOT/third_party/Megatron-Bridge"
uv pip install --python "$PYTHON" --no-deps --no-build-isolation "$ROOT/third_party/Megatron-Bridge"
# This release wheel differs from the registry build despite sharing version
# 0.3.2. Pin its content hash so the binary input cannot change unnoticed.
uv pip install --python "$PYTHON" --reinstall --no-deps \
  "$GITHUB_RELEASE_PROXY/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=c26d31b9decd4bef04c5c2bf72470e7660c2802b9599b885eb7c1dfbfbd9a4b5"

# Slime itself is pinned and carries the repository's SAO patch.
apply_patch_once "$SLIME_DIR" 1 "$ROOT/patches/slime_sao.patch"
uv pip install --python "$PYTHON" --no-deps --editable "$SLIME_DIR"
uv pip install --python "$PYTHON" --no-deps --no-build-isolation \
  "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"

# The slime image applies four SGLang source patches. Apply them to the wheel's
# Python package; -p2 maps a/python/sglang/... to site-packages/sglang/....
SGLANG_SITE="$($PYTHON -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).parent.parent)')"
SGLANG_PATCHES=(
  "$SLIME_DIR/docker/patch/latest/sglang.patch"
  "$SLIME_DIR/docker/patch/latest/sglang-top_p.patch"
  "$SLIME_DIR/docker/patch/latest/sglang-release_hicache.patch"
  "$SLIME_DIR/docker/patch/latest/sglang-pull_weights.patch"
)
if sglang_patch_stack_is_applied "$SGLANG_SITE" "${SGLANG_PATCHES[@]}"; then
  echo "SGLang patch stack already applied."
else
  for patch_file in "${SGLANG_PATCHES[@]}"; do
    apply_patch_once "$SGLANG_SITE" 2 "$patch_file"
  done
fi

# Optional FA3 source build. FA2 is sufficient for this H20 profile; enable this
# only when bit-for-bit parity with the Docker image is required.
if [[ "${INSTALL_FA3:-0}" == "1" ]]; then
  fetch_archive Dao-AILab/flash-attention "$FA3_COMMIT" "$ROOT/third_party/flash-attention-fa3"
  FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --python "$PYTHON" --no-deps --no-build-isolation \
    "$ROOT/third_party/flash-attention-fa3/hopper"
fi

"$ROOT/scripts/preflight.sh"
