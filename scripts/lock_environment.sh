#!/usr/bin/env bash
# Resolve every registry dependency to an exact version for the cu130 runtime.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null
cd "$ROOT"

uv pip compile requirements/runtime-cu130.in \
  --override requirements/runtime-cu130.overrides.txt \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_34 \
  --index-strategy unsafe-best-match \
  --default-index "${SAO_LOCK_INDEX:-https://pypi.org/simple}" \
  --extra-index-url https://pypi.nvidia.com \
  --find-links https://tile-ai.github.io/whl/nightly/cu128/ \
  --emit-find-links \
  --no-emit-package flash-attn-4 \
  --no-emit-package torch-memory-saver \
  --no-header \
  --generate-hashes \
  --output-file requirements/runtime-cu130.lock.txt
