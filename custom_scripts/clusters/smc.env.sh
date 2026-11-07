# SMC (Slurm). Sourced by custom_scripts/common/services.sh when CLUSTER_NAME=smc.
# Must set LOCAL_IP and MOONCAKE_MASTER_SERVER_ADDRESS.
#
# GPU nodes have Mellanox InfiniBand NICs; Mooncake uses rdma (see smc.yaml).

module load cuda/13.0

# LOCAL_IP may be preset, e.g. for a DRY_RUN outside a job.
LOCAL_IP="${LOCAL_IP:-$(hostname -I | awk '{print $1}')}"
export MOONCAKE_MASTER_SERVER_ADDRESS="${MASTER_ADDR}"

# SMC launches never pre-read the dataset (Weka); keep it that way until measured.
PREREAD_TRAIN_DATA=0
