#!/usr/bin/env bash
# Multi-node torchrun launcher, run identically on every pod by
# `runai submit-dist pytorch` (via csub.py --workers). RunAI injects
# PET_NNODES / RUNAI_NUM_OF_GPUS / RANK / MASTER_ADDR / MASTER_PORT per pod;
# this script turns those into a torchrun invocation so every spawned process
# gets its own real RANK/LOCAL_RANK/WORLD_SIZE (see utils/distributed_utils.py).
#
# Usage (all args after the script path are forwarded to parallel_experiments.py;
# --config is also required so the dataset cache can be built before training):
#   ./run_multinode.sh --config configs/gpt2_ddp.yaml --profile final_presentation
set -euo pipefail

echo "START TIME: $(date)"
echo "Role: $(hostname -s | tr -dc '0-9')"
echo "Num nodes: $PET_NNODES"
echo "GPUs/node: $RUNAI_NUM_OF_GPUS"
echo "rdzv endpoint: $MASTER_ADDR:$MASTER_PORT"

# Pull out --config's value: needed standalone to build the dataset cache
# below, in addition to being forwarded to parallel_experiments.py as-is.
CONFIG_PATH=""
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--config" ]; then
        CONFIG_PATH="${args[$((i + 1))]}"
        break
    fi
done
if [ -z "$CONFIG_PATH" ]; then
    echo "run_multinode.sh: --config <path> is required" >&2
    exit 1
fi

# RunAI runs this identical command on every pod, but build_dataset_cache.py
# isn't safe for concurrent multi-pod writes to the same cache dir -- only
# rank 0 builds it. The other pods don't need an explicit wait: they reach
# torchrun's rendezvous below immediately and simply block there until rank
# 0 (which builds the cache first) reaches its own torchrun call, so no
# process starts training against a half-written cache.
if [ "$RANK" = "0" ]; then
    python build_dataset_cache.py --config "$CONFIG_PATH" --workers 32 --tasks 128
fi

# RDMA for efficient inter-node NCCL comms -- see docs/multinode.md in
# getting-started for why these are needed across nodes (not just single-node).
export NCCL_IB_GID_INDEX=$(grep 'RoCE v2' $(grep '0000:0000:0000:0000:0000:ffff' /sys/class/infiniband/mlx5_bond_0/ports/1/gids/* | cut -d ':' -f 1 | sed 's/gids/gid_attrs\/types/') | sed -e 's/.*\/\([0-9]*\):.*/\1/')
export NCCL_IB_HCA=mlx5_bond_
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=$RUNAI_NUM_OF_GPUS

python -m torch.distributed.run \
    --nnodes="$PET_NNODES" \
    --nproc-per-node="$RUNAI_NUM_OF_GPUS" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --role "$(hostname -s | tr -dc '0-9')": \
    --max-restarts=0 \
    --tee 1 \
    parallel_experiments.py "$@"
