#!/bin/bash
# Script used to run custom TorchSpec code. Adapted from examples/qwen3-8b-single-node/run.sh
#
#SBATCH --job-name=torchspec_infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --exclude=smc-pod-4
#SBATCH --ignore-pbs
#SBATCH --output=/mnt/weka/aisg/users/karthik/model_training_team/TorchSpec/logs/%j.log

#PBS -N torchspec_infer
# mem must be explicit: the PBS server default is 10mb, and the Mooncake
# segment alone pins 256GB of host memory.
#PBS -l select=1:ngpus=4:ncpus=64:mem=950gb
#PBS -l walltime=24:00:00
#PBS -k oed
#PBS -j oe
#PBS -P 12004500
#PBS -q normal
#PBS -o /data/projects/12004500/karthik/TorchSpec/logs/

source ~/.bashrc
set -x

# Cluster-specific setup (network, Mooncake transport, modules) comes from
# custom_scripts/clusters/${CLUSTER_NAME}.{env.sh,yaml}; the shared startup is in
# custom_scripts/common/services.sh. DRY_RUN=1 prints the resolved setup and exits.
source "${BASE_DIR}/TorchSpec/custom_scripts/common/services.sh" || exit 1

ts_init_job
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/cache/compiled_kernels"
ts_load_cluster "${1:-${TS_ROOT}/custom_scripts/vllm_nemotron_3_super_120b.yaml}"
ts_start_services

# The aux hidden-state layer ID convention behind --speculative-config below
# (the +1 shift and the trailing num_hidden_layers slot) is explained in
# custom_scripts/README.md. Read it before changing those ids.
${BASE_DIR}/uv_biome/torchspec/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model aisingapore/Nemotron-SEA-LION-v5-120B-A12B \
    --max-model-len 16385 \
    --load-format instanttensor \
    --gpu-memory-utilization 0.85 \
    --port 8080 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --max-num-batched-tokens 65536 \
    --enable_chunked_prefill \
    --enforce-eager \
    --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [2, 13, 23, 33, 44, 55, 65, 75, 86, 88]}}}' \
    --kv-transfer-config '{"kv_connector": "MooncakeHiddenStatesConnector", "kv_connector_module_path": "torchspec.inference.engine.mooncake_hidden_states_connector", "kv_role": "kv_producer"}' &

VLLM_PID=$!
ts_track_pid "${VLLM_PID}"
ts_wait_for_vllm "${VLLM_PID}"
ts_preread_train_data

ts_run_train_entry
