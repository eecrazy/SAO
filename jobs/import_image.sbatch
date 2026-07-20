#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=nvr_lacr_llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao/scratch_impl/logs/%x_%j.out

# paper_006_sao — import the slime Docker image to .sqsh on a compute node.
# Login node cannot: / is 100% full and lustre forbids overlayfs whiteouts
# (mknod), so extraction temp must be node-local (/raid, the cluster default).
# Layer cache stays on lustre (survives across attempts/nodes).
set -euxo pipefail

IMG_DIR=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images
B=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/enroot_tmp
OUT=$IMG_DIR/slime_nightly-dev-20260707a.sqsh
[ -s "$OUT" ] && { echo "already imported"; ls -lh "$OUT"; exit 0; }

df -h /raid || df -h /
mkdir -p "$B/cache"

cd "$IMG_DIR"
ENROOT_CACHE_PATH="$B/cache" \
enroot import -o "$OUT" docker://slimerl/slime:nightly-dev-20260707a

ls -lh "$OUT"
echo IMPORT_OK
