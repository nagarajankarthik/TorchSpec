#!/bin/bash
# Script used to run custom TorchSpec code. Adapted from examples/qwen3-8b-single-node/run.sh
#
#SBATCH --job-name=torchspec_infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --ignore-pbs

#PBS -N torchspec_infer
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q AISG_debug
#PBS -o /scratch_aisg/scratch_aisg/karthik/model_training_team/TorchSpec/logs/

source ~/.bashrc
set -euo pipefail
set -x

if [[ -n "${SLURM_JOB_ID:-}" ]] ; then
    export JOB_ID=${SLURM_JOB_ID}
    export JOB_NAME=${SLURM_JOB_NAME}
    hosts=$(scontrol show hostnames ${SLURM_JOB_NODELIST})
    export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
    export NUM_NODES=${SLURM_JOB_NUM_NODES}
    echo "Running on SLURM cluster."
elif [[ -n "${PBS_JOBID:-}" ]] ; then
    export JOB_ID=${PBS_JOBID}
    export JOB_NAME=${PBS_JOBNAME}
    hosts=$(cat $PBS_NODEFILE)
    export MASTER_ADDR=$(cat $PBS_NODEFILE | head -n 1)
    export NUM_NODES=$(cat $PBS_NODEFILE | wc -l)
    echo "Running on PBS cluster."
fi

export GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/cache/compiled_kernels"
export TORCHSPEC_LOG_LEVEL=DEBUG

cd ${BASE_DIR}/TorchSpec
export PYTHONPATH=${BASE_DIR}/TorchSpec
module load cuda/13.0

CONFIG_FILE="${1:-${BASE_DIR}/TorchSpec/custom_scripts/vllm_nemotron_3_super_120b.yaml}"

export LOG_DIR="${BASE_DIR}/TorchSpec/logs/${JOB_ID}"
mkdir -p $LOG_DIR
export MOONCAKE_MASTER_SERVER_ADDRESS="${MASTER_ADDR}"
export MOONCAKE_ENV_FILE="${LOG_DIR}/mooncake_env.sh"

LOCAL_IP=$(hostname -I | awk '{print $1}')
if [[ -z "${LOCAL_IP}" ]]; then
    echo "ERROR: could not determine a local IP from 'hostname -I'" >&2
    exit 1
fi

# Export Mooncake environment variables read by vllm connector.
${BASE_DIR}/uv_biome/torchspec/bin/python3 -m torchspec.mooncake_helper \
    --config ${CONFIG_FILE} \
    mooncake.env_file=${MOONCAKE_ENV_FILE} \
    mooncake.local_hostname=${LOCAL_IP}

# Launch mooncake master server. Port numbers here must be consistent with the ones in the config file.
MOONCAKE_MASTER_PATH=${BASE_DIR}/uv_biome/torchspec/lib/python3.12/site-packages/mooncake/mooncake_master
${MOONCAKE_MASTER_PATH} \
  --port="8011" \
  --http_metadata_server_port="8012" \
  --http_metadata_server_host=0.0.0.0 \
  --enable_http_metadata_server=true \
  --default_kv_lease_ttl=5000 \
  --metrics_port="8013" & MC_PID=$!

trap "kill -TERM $MC_PID 2>/dev/null || true" EXIT
until timeout 1 bash -c '</dev/tcp/localhost/8011' && timeout 1 bash -c '</dev/tcp/localhost/8012'; do
    kill -0 $MC_PID 2>/dev/null || { echo "mooncake_master died"; exit 1; }
    sleep 1
done

# 1. Launch vLLM in the background
# The following comment block in torchspec/inference/engine/vllm_engine should be noted:
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
source ${MOONCAKE_ENV_FILE}  
${BASE_DIR}/uv_biome/torchspec/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --port 8080 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --max-num-batched-tokens 65536 \
    --enable-chunked-prefill \
    --enforce-eager \
    --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [5, 30, 60, 88]}}}' \
    --kv-transfer-config '{"kv_connector": "MooncakeHiddenStatesConnector", "kv_connector_module_path": "torchspec.inference.engine.mooncake_hidden_states_connector", "kv_role": "kv_producer"}' &

VLLM_PID=$!

trap "kill -TERM $MC_PID $VLLM_PID 2>/dev/null || true" EXIT

# 2. Wait until the vLLM endpoint is live and healthy
echo "Waiting for vLLM server to start..."
export VLLM_STARTUP_TIMEOUT=7200
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-3600}"
VLLM_DEADLINE=$(( SECONDS + VLLM_STARTUP_TIMEOUT ))
until curl -s http://localhost:8080/health > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server failed to start!" >&2
        exit 1
    fi
    if (( SECONDS > VLLM_DEADLINE )); then
        # A hung worker keeps the parent PID alive forever, so without this the
        # job sits here until walltime. Check the VLLM::Worker_* logs.
        echo "ERROR: vLLM not healthy after ${VLLM_STARTUP_TIMEOUT}s; check worker logs." >&2
        exit 1
    fi
    sleep 5
done
echo "vLLM server is ready!"


${BASE_DIR}/uv_biome/torchspec/bin/python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    mooncake.local_hostname=${LOCAL_IP}
