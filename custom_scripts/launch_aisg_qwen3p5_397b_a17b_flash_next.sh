#!/bin/bash
#SBATCH --job-name=torchspec_infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
##SBATCH --exclude=smc-pod-4
#SBATCH --ignore-pbs
#SBATCH --output=/mnt/weka/aisg/users/karthik/model_training_team/TorchSpec/logs/%j.log

#PBS -N torchspec_infer
# mem must be explicit: the PBS server default is 10mb, and the Mooncake
# segment alone pins 256GB of host memory.
#PBS -l select=1:ngpus=8:ncpus=128:mem=1900gb
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
ts_load_cluster "${1:-${TS_ROOT}/custom_scripts/vllm_aisg_qwen3p5_397b_flash_next.yaml}"
ts_start_services

# KV-cache geometry, the prefix-caching requirement, the aux hidden-state layer
# ID convention and the torch.compile staging plan are all explained in
# custom_scripts/README.md. Read it before changing any flag below.
#
# Exactly one COMPILE_ARGS line should be uncommented. Validate hidden states
# against an eager run before advancing a stage.

# Stage 1 (active): inductor only, no graph capture.
# COMPILE_ARGS=(-cc.mode=3 -cc.cudagraph_mode=NONE)

# Stage 2: add piecewise graphs; attention stays outside the graph.
# COMPILE_ARGS=(-cc.mode=3 -cc.cudagraph_mode=PIECEWISE -cc.cudagraph_capture_sizes='[1,2,4,8,16]')

# Stage 3: vllm's v1 default -- full graphs for decode, piecewise for prefill.
COMPILE_ARGS=(-cc.mode=3 -cc.cudagraph_mode=FULL_AND_PIECEWISE -cc.cudagraph_capture_sizes='[1,2,4,8,16]')

# Rollback to the previous behaviour:
# COMPILE_ARGS=(--enforce-eager)

export INSTANTTENSOR_BUFFER_SIZE=8589934592
${BASE_DIR}/uv_biome/vllm_custom/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model aisingapore/Qwen3.5-397B-A17B-FlashNext-Prewarm-Exp-2 \
    --max-model-len 16385 \
    --load-format instanttensor \
    --gpu-memory-utilization 0.90 \
    --port 8080 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    "${COMPILE_ARGS[@]}" \
    --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [2, 34, 58, 60]}}}' \
    --kv-transfer-config '{"kv_connector": "MooncakeHiddenStatesConnector", "kv_connector_module_path": "torchspec.inference.engine.mooncake_hidden_states_connector", "kv_role": "kv_producer"}' &

VLLM_PID=$!
ts_track_pid "${VLLM_PID}"
ts_wait_for_vllm "${VLLM_PID}"
ts_preread_train_data

# resume_generation=true: a job killed at walltime is resubmitted with this same
# script and continues from <output_dir>/generation_progress.json, skipping
# rows already published. With no progress file it starts from epoch 0. To
# start over deliberately, move that file away first.
ts_run_train_entry training.resume_generation=true
