#!/bin/bash
#SBATCH --partition=batch_block1
#SBATCH --account=nvr_lacr_llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao/scratch_impl/logs/%x_%j.out

# paper_006_sao — gate S-1: container sanity on a GPU node. Verifies the
# enroot-imported slime image: imports (slime/megatron/sglang/torch/ray + our
# sao_plugin), CUDA visibility, and a 30-second single-GPU sglang server
# round-trip on Qwen3.5-4B.
set -euxo pipefail

CONTAINER=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/slime_nightly-dev-20260707a.sqsh
SB=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/AI-Reproduction/sandboxes/paper_006_sao
CODE=$SB/scratch_impl/code
export HF_HOME=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/huggingface
HF_MODEL=$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home \
     --no-container-entrypoint \
     bash -c "set -eux
        nvidia-smi -L
        export PYTHONPATH=/root/Megatron-LM:$CODE/slime:$CODE
        export HF_HOME=$HF_HOME
        python - << 'EOF'
import torch, ray, sglang, transformers
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())
print('sglang', sglang.__version__, '| transformers', transformers.__version__)
import megatron.core as mc; print('megatron.core OK')
import slime; print('slime OK (pinned clone shadows pip)')
import slime.utils.arguments  # full arg tree parses
from sao_plugin import dis_loss, adaptive_gae, reward_math, async_staleness, data_prep, filter_campaign
print('sao_plugin imports OK')
assert reward_math.grade_response(r'the answer is \\boxed{42}', '42') == 1
assert reward_math.grade_response(r'blah \\boxed{41}', '42') == 0
assert reward_math.grade_response(r'<think>x</think> \\boxed{7}', '7') == 1
print('grade_response sanity OK')
EOF
        # 30s sglang round-trip (1 GPU) on the actual model
        python -m sglang.launch_server --model-path $HF_MODEL \
           --served-model-name qwen3.5-4b --tp-size 1 --mem-fraction-static 0.8 \
           --host 127.0.0.1 --port 30001 --log-level warning &
        SPID=\$!
        for i in \$(seq 60); do curl -sf http://127.0.0.1:30001/health >/dev/null && break; sleep 5; done
        curl -sf http://127.0.0.1:30001/v1/chat/completions -H 'Content-Type: application/json' -d '{
          \"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer with \\\\boxed{}.\"}],
          \"max_tokens\":512,\"temperature\":1.0}' | head -c 2000
        kill \$SPID 2>/dev/null || true
        echo; echo S-1_CONTAINER_SMOKE_OK"
