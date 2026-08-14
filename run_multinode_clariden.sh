#!/usr/bin/env bash
# Multi-node launcher for Clariden (CSCS Alps, GH200/SLURM), mirroring
# run_multinode.sh's RunAI/RCP flow but adapted to SLURM idioms: one srun
# task per GPU gives each process its own RANK/LOCAL_RANK directly from
# Slurm, so there's no need for torchrun's per-node fan-out.
#
# Usage (all args after the script path are forwarded to
# parallel_experiments.py; --config is also required so the dataset cache
# can be built before training):
#   sbatch --nodes=2 run_multinode_clariden.sh \
#       --config configs/gpt2-ddp-multinode.yaml --profile final-presentation
#
#SBATCH --account=infra01
#SBATCH --job-name=curriculum-multinode
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --nodes=2
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail

CONTAINER_ENV="$HOME/curriculum-learning-using-attention/clariden.toml"

echo "START TIME: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES  Tasks/node: $SLURM_NTASKS_PER_NODE"

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
    echo "run_multinode_clariden.sh: --config <path> is required" >&2
    exit 1
fi

# Dataset cache build: single task, CPU-only. Uses a separate venv
# (curriculum-datacache-venv) because datatrove requires numpy>=2, which is
# incompatible with the training venv's prebuilt CUDA torch (numpy<2 ABI).
srun --nodes=1 --ntasks=1 --environment="$CONTAINER_ENV" bash -c '
    source /iopsstor/scratch/cscs/$USER/curriculum-datacache-venv/bin/activate
    python build_dataset_cache.py --config "'"$CONFIG_PATH"'" --workers 32 --tasks 128
'

# Main distributed training: one Slurm task per GPU across all nodes, so
# RANK/LOCAL_RANK come straight from SLURM_PROCID/SLURM_LOCALID -- no
# torchrun rendezvous needed. MASTER_ADDR/PORT/WORLD_SIZE are exported here
# and inherited by every srun task.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS

srun --mpi=pmix --environment="$CONTAINER_ENV" --network=disable_rdzv_get \
    bash -c '
        source /iopsstor/scratch/cscs/$USER/curriculum-venv/bin/activate
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=$SLURM_LOCALID
        exec python parallel_experiments.py "$@"
    ' bash "$@"

echo "END TIME: $(date)"
