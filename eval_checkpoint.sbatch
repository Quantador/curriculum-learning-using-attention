#!/usr/bin/env bash
# Re-score one trained checkpoint on the OPUS-comparable suite, on the debug partition.
#
#   sbatch [--job-name=<label>] eval_checkpoint.sbatch <run_dir_name> [experiment_name]
# e.g.
#   sbatch --job-name=eval-router eval_checkpoint.sbatch 20260820_110508_experiment_baseline
#   sbatch --job-name=eval-random eval_checkpoint.sbatch 20260820_110508_random_batch_baseline
#
# experiment_name defaults to <run_dir_name> with its leading YYYYMMDD_HHMMSS_ stripped,
# which is how parallel_experiments.py builds those directory names in the first place.
# Passing --job-name is worth it: both arms otherwise share one name and the only thing
# telling their logs apart is the job id.
#
# Results land in the RUN'S OWN directory:
#   results/_parallel_run/<run_dir_name>/eval_scores.json
# not results/<experiment_name>/. experiment_name is not unique across runs -- every sweep
# emits an "experiment_baseline" and a "random_batch_baseline" -- so a name-keyed path is a
# single shared slot that later runs silently overwrite. The timestamped dir already
# isolates each run's configs/, logs/, status/ and checkpoints/; scores belong beside them.
# compare_to_opus.py takes run directories positionally, so it reads these paths directly.
#
# ONE GPU, not 4x4. evaluate_checkpoint.py is single-process: no DDP, no torchrun -- it
# just does model.to(cfg.device) and hands the model to lm-eval's HFLM. GPT-2 XL is ~6GB,
# so one GPU is ample, and the end-of-training eval inside the 16-GPU training jobs was
# already rank-0-only (utils/experiment_worker.py's _end_of_training_eval), i.e. also
# single-GPU. Nodes here are OverSubscribe=EXCLUSIVE so this still gets a whole node;
# ntasks-per-node=1 just avoids launching 4 redundant ranks that would each load the model
# and race to write the same output file.
#
# debug-partition limits this is sized against (scontrol show partition debug /
# sacctmgr show qos debug-qos):
#   MaxNodes=4, MaxTime=01:30:00, MaxTRESMins=node=90, MaxJobsPU=1, MaxSubmitPU=2
# At 1 node, 90 minutes is simultaneously the wall-clock cap AND the entire 90 node-minute
# budget, so 01:30:00 is the most debug can give this job. Submitting both arms is fine
# (MaxSubmitPU=2), but MaxJobsPU=1 means they run one after the other, not concurrently.
#SBATCH --account=infra01
#SBATCH --job-name=eval-ckpt
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --no-requeue
set -euo pipefail

RUN_DIR="${1:?usage: sbatch eval_checkpoint.sbatch <run_dir_name> [experiment_name]}"
EXP_NAME="${2:-$(sed -E 's/^[0-9]{8}_[0-9]{6}_//' <<<"$RUN_DIR")}"

REPO=/curriculum-learning-using-attention
VENV=/iopsstor/scratch/cscs/$USER/curriculum-venv
RUN_REL="results/_parallel_run/$RUN_DIR"

echo "run dir    : $RUN_REL"
echo "checkpoint : $RUN_REL/checkpoints/$EXP_NAME.pt"
echo "scores out : $RUN_REL/eval_scores.json"

srun --environment="$SLURM_SUBMIT_DIR/clariden.toml" bash -c "
    set -euo pipefail
    source $VENV/bin/activate
    cd $REPO
    python evaluate_checkpoint.py \
        --checkpoint '$RUN_REL/checkpoints/$EXP_NAME.pt' \
        --config configs/own_embeddings_ppo_vs_random.yaml \
        --out '$RUN_REL/eval_scores.json'
"
