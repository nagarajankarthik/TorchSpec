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
export TORCHSPEC_LOG_LEVEL=DEBUG

cd ${BASE_DIR}/TorchSpec
export PYTHONPATH=${BASE_DIR}/TorchSpec
module load cuda/13.0

CONFIG_FILE="${1:-${BASE_DIR}/TorchSpec/custom_scripts/vllm_aisg_qwen3p5_397b_flash_next.yaml}"

export LOG_DIR="${BASE_DIR}/TorchSpec/logs/${JOB_ID}"
mkdir -p $LOG_DIR
export MOONCAKE_ENV_FILE="${BASE_DIR}/TorchSpec/logs/${JOB_ID}/mooncake_env.sh"

# Aspire 2B. Also set protocol to tcp and remove device_name in mooncake section of config file.
# GPU nodes are Slingshot-only; 'hostname -I' and the node hostname both resolve
# to bond0, a 1GbE management port, so advertise hsn0 instead.
LOCAL_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
if [[ -z "${LOCAL_IP}" ]]; then
    echo "ERROR: could not determine an IPv4 address on hsn0" >&2
    exit 1
fi
# Single-node job: the master runs here, so reach it over hsn0 too.
export MOONCAKE_MASTER_SERVER_ADDRESS="${LOCAL_IP}"

# Redis is installed per-user and may not be on PATH; override with
# REDIS_SERVER=/path/to/redis-server. Checked before anything is started.
REDIS_SERVER="${REDIS_SERVER:-$(command -v redis-server)}"
if [[ ! -x "${REDIS_SERVER}" ]]; then
    echo "ERROR: redis-server not found; set REDIS_SERVER or add it to PATH" >&2
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

# Launch Redis server
"${REDIS_SERVER}" --port 6390 --bind "${LOCAL_IP}" 127.0.0.1 --protected-mode no \
             --save '' --appendonly no \
             --maxmemory 2gb --maxmemory-policy noeviction &
REDIS_PID=$!
trap "kill -TERM $MC_PID $REDIS_PID 2>/dev/null || true" EXIT
until timeout 1 bash -c "</dev/tcp/localhost/6390"; do
    kill -0 $REDIS_PID 2>/dev/null || { echo "redis-server died" >&2; exit 1; }
    sleep 1
done

# Host and port are deployment facts, so they are set here. The stream names
# are NOT: mooncake_helper already wrote them into this file from the config,
# which is what the controller reads. Duplicating them here would let the two
# drift, and the producer would publish to one stream while consumers blocked
# on another -- no error, no data.
echo "" >> "${MOONCAKE_ENV_FILE}"
cat >> "${MOONCAKE_ENV_FILE}" <<EOF
export REDIS_HOST=${LOCAL_IP}
export REDIS_PORT=6390
EOF

export MOONCAKE_VERIFY_PUTS=0

# 1. Launch vLLM in the background
source ${MOONCAKE_ENV_FILE}

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
    --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [2, 10, 18, 26, 34, 42, 50, 58, 60]}}}' \
    --kv-transfer-config '{"kv_connector": "MooncakeHiddenStatesConnector", "kv_connector_module_path": "torchspec.inference.engine.mooncake_hidden_states_connector", "kv_role": "kv_producer"}' &

VLLM_PID=$!

trap "kill -TERM $MC_PID $REDIS_PID $VLLM_PID 2>/dev/null || true" EXIT

# 2. Wait until the vLLM endpoint is live and healthy
echo "Waiting for vLLM server to start..."
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-7200}"
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

# Pre-read the training data into the page cache. /data is Lustre, and the
# dataset is mmapped by load_from_disk, so the loader otherwise pulls it in
# through small page faults (~300 rows/s, hours for 2.2M rows). Large parallel
# sequential reads run at full Lustre bandwidth and make the loader ~30x faster.
# This runs after vLLM is up so it does not compete with the weight load.
TRAIN_DATA_PATH=$(${BASE_DIR}/uv_biome/torchspec/bin/python3 -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["dataset"]["train_data_path"] or "")' \
    "$CONFIG_FILE")
if [[ -d "${TRAIN_DATA_PATH}" ]]; then
    echo "Pre-reading ${TRAIN_DATA_PATH} into page cache..."
    PREREAD_START=${SECONDS}
    find "${TRAIN_DATA_PATH}" -name '*.arrow' -print0 | xargs -0 -P 16 -n 8 cat > /dev/null
    echo "Pre-read finished in $(( SECONDS - PREREAD_START ))s."
else
    echo "WARNING: train_data_path '${TRAIN_DATA_PATH}' is not a local directory; skipping pre-read." >&2
fi


# resume_generation=true: a job killed at walltime is resubmitted with this same
# script and continues from <output_dir>/generation_progress.json, skipping
# rows already published. With no progress file it starts from epoch 0. To
# start over deliberately, move that file away first.
${BASE_DIR}/uv_biome/torchspec/bin/python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    mooncake.local_hostname=${LOCAL_IP} \
    training.resume_generation=true
