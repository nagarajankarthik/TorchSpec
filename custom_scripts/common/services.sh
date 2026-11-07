# Shared startup for custom_scripts/launch_*.sh. Source it; do not execute it.
#
# A launch script keeps only what belongs to its model (the vllm command, TP
# size, aux layer ids, compile stage) and calls these in order:
#
#   ts_init_job                    # JOB_ID, LOG_DIR, cd, PYTHONPATH
#   ts_load_cluster "$MODEL_CONFIG"   # sources clusters/${CLUSTER_NAME}.env.sh
#   ts_start_services              # Mooncake env + master, Redis. DRY_RUN=1 stops here.
#   <start vllm in the background>
#   ts_track_pid $!; ts_wait_for_vllm $!
#   ts_preread_train_data          # only if the cluster asks for it
#   ts_run_train_entry [overrides...]
#
# Everything that differs between clusters lives in clusters/<name>.env.sh
# (network, transport env vars, modules) and clusters/<name>.yaml (config keys
# merged after the model config). Nothing in here should name a cluster.

: "${BASE_DIR:?BASE_DIR is not set; export it in ~/.bashrc}"
TS_ROOT="${BASE_DIR}/TorchSpec"
TS_PY="${BASE_DIR}/uv_biome/torchspec/bin/python3"
TS_CLUSTER_DIR="${TS_ROOT}/custom_scripts/clusters"
TS_MOONCAKE_MASTER="${BASE_DIR}/uv_biome/torchspec/lib/python3.12/site-packages/mooncake/mooncake_master"

# Ports must match mooncake.{master,metadata,metrics}_port in the model config
# and redis_port is what consumers are told via MOONCAKE_ENV_FILE.
TS_MOONCAKE_MASTER_PORT=8011
TS_MOONCAKE_METADATA_PORT=8012
TS_MOONCAKE_METRICS_PORT=8013
TS_REDIS_PORT=6390
# Read lease on a fetched object. A read slower than this fails with
# LEASE_EXPIRED, so it bounds the slowest acceptable consumer fetch.
TS_MOONCAKE_LEASE_TTL_MS="${TS_MOONCAKE_LEASE_TTL_MS:-5000}"

TS_PIDS=()

ts_die() {
    echo "ERROR: $*" >&2
    exit 1
}

_ts_cleanup() {
    if ((${#TS_PIDS[@]})); then
        kill -TERM "${TS_PIDS[@]}" 2>/dev/null || true
    fi
}

# Every background service is registered here so one EXIT trap stops them all.
ts_track_pid() {
    TS_PIDS+=("$1")
    trap _ts_cleanup EXIT
}

ts_init_job() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        export JOB_ID=${SLURM_JOB_ID}
        export JOB_NAME=${SLURM_JOB_NAME}
        export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
        export NUM_NODES=${SLURM_JOB_NUM_NODES}
    elif [[ -n "${PBS_JOBID:-}" ]]; then
        export JOB_ID=${PBS_JOBID}
        export JOB_NAME=${PBS_JOBNAME}
        export MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}")
        export NUM_NODES=$(wc -l < "${PBS_NODEFILE}")
    else
        # Outside a scheduler: only useful with DRY_RUN=1.
        export JOB_ID=${JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}
        export JOB_NAME=${JOB_NAME:-local}
        export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
        export NUM_NODES=1
    fi
    export GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-DEBUG}"

    cd "${TS_ROOT}" || ts_die "cannot cd to ${TS_ROOT}"
    export PYTHONPATH="${TS_ROOT}"
    export LOG_DIR="${TS_ROOT}/logs/${JOB_ID}"
    [[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p "${LOG_DIR}"
    export MOONCAKE_ENV_FILE="${LOG_DIR}/mooncake_env.sh"
}

# Fails before anything starts if CLUSTER_NAME is missing or unknown: a wrong
# cluster file does not error later, it just picks the wrong network.
ts_load_cluster() {
    MODEL_CONFIG="$1"
    [[ -f "${MODEL_CONFIG}" ]] || ts_die "model config not found: ${MODEL_CONFIG}"

    local known
    known=$(cd "${TS_CLUSTER_DIR}" && ls *.env.sh 2>/dev/null | sed 's/\.env\.sh$//' | tr '\n' ' ')
    [[ -n "${CLUSTER_NAME:-}" ]] ||
        ts_die "CLUSTER_NAME is not set. Add 'export CLUSTER_NAME=<name>' to ~/.bashrc. Known: ${known}"
    local env_file="${TS_CLUSTER_DIR}/${CLUSTER_NAME}.env.sh"
    CLUSTER_CONFIG="${TS_CLUSTER_DIR}/${CLUSTER_NAME}.yaml"
    [[ -f "${env_file}" && -f "${CLUSTER_CONFIG}" ]] ||
        ts_die "CLUSTER_NAME=${CLUSTER_NAME} needs ${env_file} and ${CLUSTER_CONFIG}. Known: ${known}"

    # shellcheck source=/dev/null
    source "${env_file}"

    [[ -n "${LOCAL_IP:-}" ]] || ts_die "${env_file} did not set LOCAL_IP"
    [[ -n "${MOONCAKE_MASTER_SERVER_ADDRESS:-}" ]] ||
        ts_die "${env_file} did not set MOONCAKE_MASTER_SERVER_ADDRESS"
    # Redis is installed per user and may not be on PATH.
    REDIS_SERVER="${REDIS_SERVER:-$(command -v redis-server)}"
    [[ -x "${REDIS_SERVER}" ]] || ts_die "redis-server not found; set REDIS_SERVER or add it to PATH"

    # Model config first, cluster config second: the cluster file only sets
    # keys that differ between clusters (e.g. mooncake.protocol).
    TS_CONFIG_ARGS=(--config "${MODEL_CONFIG}" --config "${CLUSTER_CONFIG}")
}

ts_print_setup() {
    echo "===== ${CLUSTER_NAME}: resolved launch setup"
    echo "JOB_ID=${JOB_ID}  LOCAL_IP=${LOCAL_IP}  MOONCAKE_MASTER_SERVER_ADDRESS=${MOONCAKE_MASTER_SERVER_ADDRESS}"
    echo "REDIS_SERVER=${REDIS_SERVER}  PREREAD_TRAIN_DATA=${PREREAD_TRAIN_DATA:-0}"
    env | grep -E '^MC_' | sort
    echo "configs: ${MODEL_CONFIG} + ${CLUSTER_CONFIG}"
    "${TS_PY}" - "${TS_CONFIG_ARGS[@]}" "mooncake.local_hostname=${LOCAL_IP}" <<'EOF'
import argparse
from omegaconf import OmegaConf
from torchspec.config.train_config import load_config

p = argparse.ArgumentParser()
p.add_argument("--config", action="append", required=True)
a, overrides = p.parse_known_args()
c = load_config(config_path=a.config, cli_args=overrides)
for key in ("dataset.train_data_path", "output_dir", "mooncake.protocol", "mooncake.device_name",
            "mooncake.local_hostname", "mooncake.num_aux_layers", "inference.aux_hidden_states_layers",
            "redis.stream_maxlen"):
    print(f"{key} = {OmegaConf.select(c, key)}")
EOF
}

ts_start_services() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        ts_print_setup
        echo "DRY_RUN=1: nothing started."
        exit 0
    fi

    # Mooncake env read by the vLLM connector and by consumers.
    "${TS_PY}" -m torchspec.mooncake_helper "${TS_CONFIG_ARGS[@]}" \
        "mooncake.env_file=${MOONCAKE_ENV_FILE}" \
        "mooncake.local_hostname=${LOCAL_IP}" || ts_die "mooncake_helper failed"

    [[ -x "${TS_MOONCAKE_MASTER}" ]] || ts_die "mooncake_master not found at ${TS_MOONCAKE_MASTER}"
    "${TS_MOONCAKE_MASTER}" \
        --port="${TS_MOONCAKE_MASTER_PORT}" \
        --http_metadata_server_port="${TS_MOONCAKE_METADATA_PORT}" \
        --http_metadata_server_host=0.0.0.0 \
        --enable_http_metadata_server=true \
        --default_kv_lease_ttl="${TS_MOONCAKE_LEASE_TTL_MS}" \
        --metrics_port="${TS_MOONCAKE_METRICS_PORT}" &
    local mc_pid=$!
    ts_track_pid "${mc_pid}"
    until timeout 1 bash -c "</dev/tcp/localhost/${TS_MOONCAKE_MASTER_PORT}" &&
        timeout 1 bash -c "</dev/tcp/localhost/${TS_MOONCAKE_METADATA_PORT}"; do
        kill -0 "${mc_pid}" 2>/dev/null || ts_die "mooncake_master died"
        sleep 1
    done

    "${REDIS_SERVER}" --port "${TS_REDIS_PORT}" --bind "${LOCAL_IP}" 127.0.0.1 --protected-mode no \
        --save '' --appendonly no \
        --maxmemory 2gb --maxmemory-policy noeviction &
    local redis_pid=$!
    ts_track_pid "${redis_pid}"
    until timeout 1 bash -c "</dev/tcp/localhost/${TS_REDIS_PORT}"; do
        kill -0 "${redis_pid}" 2>/dev/null || ts_die "redis-server died"
        sleep 1
    done

    # Host and port are deployment facts, so they are set here. The stream names
    # are NOT: mooncake_helper already wrote them into this file from the config,
    # which is what the controller reads. Duplicating them here would let the two
    # drift, and the producer would publish to one stream while consumers blocked
    # on another -- no error, no data.
    cat >> "${MOONCAKE_ENV_FILE}" <<EOF

export REDIS_HOST=${LOCAL_IP}
export REDIS_PORT=${TS_REDIS_PORT}
EOF
    export MOONCAKE_VERIFY_PUTS=0
    # shellcheck source=/dev/null
    source "${MOONCAKE_ENV_FILE}"
}

ts_wait_for_vllm() {
    local vllm_pid="$1"
    local timeout_s="${VLLM_STARTUP_TIMEOUT:-7200}"
    local deadline=$((SECONDS + timeout_s))
    echo "Waiting for vLLM server to start..."
    until curl -s http://localhost:8080/health > /dev/null; do
        kill -0 "${vllm_pid}" 2>/dev/null || ts_die "vLLM server failed to start"
        # A hung worker keeps the parent PID alive forever, so without this the
        # job sits here until walltime. Check the VLLM::Worker_* logs.
        ((SECONDS <= deadline)) || ts_die "vLLM not healthy after ${timeout_s}s; check worker logs"
        sleep 5
    done
    echo "vLLM server is ready!"
}

# On a filesystem where mmap page faults are slow (Lustre on ASPIRE 2B: ~300
# rows/s, hours for 2.2M rows), large parallel sequential reads fill the page
# cache at full bandwidth and make load_from_disk ~30x faster. Runs after vLLM
# is up so it does not compete with the weight load. Clusters opt in with
# PREREAD_TRAIN_DATA=1.
ts_preread_train_data() {
    [[ "${PREREAD_TRAIN_DATA:-0}" == 1 ]] || return 0
    local path
    # Resolved through the same loader as train_entry, so ${oc.env:BASE_DIR}
    # and cluster/CLI overrides are already applied.
    path=$("${TS_PY}" - "${TS_CONFIG_ARGS[@]}" <<'EOF'
import argparse
from torchspec.config.train_config import load_config

p = argparse.ArgumentParser()
p.add_argument("--config", action="append", required=True)
print(load_config(config_path=p.parse_args().config).dataset.train_data_path or "")
EOF
    )
    if [[ ! -d "${path}" ]]; then
        echo "WARNING: train_data_path '${path}' is not a local directory; skipping pre-read." >&2
        return 0
    fi
    echo "Pre-reading ${path} into page cache..."
    local start=${SECONDS}
    find "${path}" -name '*.arrow' -print0 | xargs -0 -P 16 -n 8 cat > /dev/null
    echo "Pre-read finished in $((SECONDS - start))s."
}

ts_run_train_entry() {
    "${TS_PY}" -m torchspec.train_entry "${TS_CONFIG_ARGS[@]}" \
        "mooncake.local_hostname=${LOCAL_IP}" "$@"
}
