# ASPIRE 2B (NSCC, PBS Pro). Sourced by custom_scripts/common/services.sh when
# CLUSTER_NAME=aspire2b. Must set LOCAL_IP and MOONCAKE_MASTER_SERVER_ADDRESS.
#
# GPU nodes: 8x H200, 4x Slingshot 400Gb NICs (Cassini 2, cxi0-3 -> hsn0-3,
# 10.180.{4,5,6,7}.x), no verbs devices. bond0 (172.25.x.x) is a 1GbE management
# port, and it is what `hostname -I` and the node hostname resolve to.

module load cuda/13.0

# Advertise hsn0. LOCAL_IP may be preset, e.g. for a DRY_RUN on the login node.
LOCAL_IP="${LOCAL_IP:-$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"
# Single-node job: the master runs here, so reach it over hsn0 too.
export MOONCAKE_MASTER_SERVER_ADDRESS="${LOCAL_IP}"

# Mooncake's TCP transport ignores local_hostname when it advertises its data
# address: it picks the first UP interface, which is bond0 (1GbE). Consumers then
# read every sample at ~0.12 GB/s and hit LEASE_EXPIRED / TRANSFER_FAIL. Pin it
# to hsn0 (~4.4 GB/s per reader). More I/O threads lift 8-way reads from ~4.9
# to ~14 GB/s. Inherited by the vLLM workers, one of which owns the segment.
# Exported here, not written to MOONCAKE_ENV_FILE: consumers must bind their own IP.
export MC_TCP_BIND_ADDRESS="${LOCAL_IP}"
export MC_NUM_TCP_IO_THREADS=8

# /data is Lustre: mmap page faults make load_from_disk ~30x slower than a
# parallel sequential pre-read into the page cache.
PREREAD_TRAIN_DATA=1
