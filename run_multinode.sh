#!/usr/bin/env bash
# Multi-node torchrun launcher, run identically on every pod by
# `runai submit-dist pytorch` (via csub.py --workers). RunAI injects
# PET_NNODES / RUNAI_NUM_OF_GPUS / RANK / MASTER_ADDR / MASTER_PORT per pod;
# this script turns those into a torchrun invocation so every spawned process
# gets its own real RANK/LOCAL_RANK/WORLD_SIZE (see utils/distributed_utils.py).
#
# Usage (all args after the script path are forwarded to parallel_experiments.py):
#   ./run_multinode.sh --profile final_presentation
set -euo pipefail

echo "START TIME: $(date)"
echo "Role: $(hostname -s | tr -dc '0-9')"
echo "Num nodes: $PET_NNODES"
echo "GPUs/node: $RUNAI_NUM_OF_GPUS"
echo "rdzv endpoint: $MASTER_ADDR:$MASTER_PORT"

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
