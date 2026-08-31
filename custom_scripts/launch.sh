#!/bin/bash
# Script used to run custom TorchSpec code. Adapted from examples/qwen3-8b-single-node/run.sh
#
# SBATCH --job-name=torchspec_infer
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:8
# SBATCH --cpus-per-gpu=16

set -euo pipefail
set -x

if [[ -n "${SLURM_JOB_ID:-}" ]] ; then
    export BASE_DIR="/mnt/weka/aisg/users/karthik/model_training_team/torchspec_test"
    export JOB_ID=${SLURM_JOB_ID}
    export JOB_NAME=${SLURM_JOB_NAME}
    hosts=$(scontrol show hostnames ${SLURM_JOB_NODELIST})
    export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
    export NUM_NODES=${SLURM_JOB_NUM_NODES}
    echo "Running on SLURM cluster."
elif [[ -n "${PBS_JOBID:-}" ]] ; then
    export BASE_DIR="/data/projects/51401024"
    export JOB_ID=${PBS_JOBID}
    export JOB_NAME=${PBS_JOBNAME}
    hosts=$(cat $PBS_NODEFILE)
    export MASTER_ADDR=$(cat $PBS_NODEFILE | head -n 1)
    export NUM_NODES=$(cat $PBS_NODEFILE | wc -l)
    echo "Running on PBS cluster."
fi

export GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export TORCHSPEC_LOG_LEVEL=INFO

CONFIG_FILE="${1:-$ROOT_DIR/custom_scripts/vllm_nemotron_3_super_120b.yaml}"

# TODO: Update the path to the actual virtual environment.

# 1. Launch vLLM in the background
# THe following comment block in torchspec/inference/engine/vllm_engine should be noted:
# Layer IDs use post-layer semantics: "capture the residual stream
# after layer N runs".  vllm's capture hook fires at the INPUT of each
# listed layer (= output of the previous layer), so we shift by +1 to
# align with sglang's convention.
# vllm's `_maybe_add_hidden_state` is called with `layer_idx + 1`
# *after* each layer runs, so valid capture indices are
# [0, num_hidden_layers]; we keep ids up to num_hidden_layers
# (the pre-`norm` slot, see final-layer block below).
# Append the model's final layer to capture last_hidden_states
# (pre-norm) for target logit computation.  Index `num_hidden_layers`
# is vllm's reserved post-last-layer / pre-`norm` slot, so training
# can apply the model's final norm itself on top of this.
/path/to/env/bin/python -m vllm.entrypoints.openai.api_server \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --port 8080 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --max-num-batched-tokens 65536 \
    --enable-chunked-prefill \
    --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [5, 30, 60, 88]}}}' \
    --kv-transfer-config '{"kv_connector": "MooncakeHiddenStatesConnector", "kv_connector_module_path": "torchspec.inference.engine.mooncake_hidden_states_connector", "kv_role": "kv_producer"}' &

VLLM_PID=$!

# Ensure vLLM is killed when the script exits or terminates
trap "kill -9 $VLLM_PID 2>/dev/null || true" EXIT

# 2. Wait until the vLLM endpoint is live and healthy
echo "Waiting for vLLM server to start..."
until curl -s http://localhost:8080/health > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server failed to start!"
        exit 1
    fi
    sleep 5
done
echo "vLLM server is ready!"




python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    "$@"


