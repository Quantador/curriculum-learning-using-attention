#!/usr/bin/env bash
# Resume the fineweb-100BT tokenization. Datatrove skips tasks that already have a
# completion marker, so this picks up the 25 unfinished ones.
#
#   sbatch build_cache.sbatch [workers]        # default 8
#
# debug partition, 1:30 (its MaxTime, and at 1 node also its whole 90 node-minute
# MaxTRESMins budget). Expect partial progress: measured over the 103 completed tasks,
# per-task duration is min 64 / p50 151 / p90 559 minutes, so only the fast tail finishes
# inside 90 minutes. Tasks that do not finish write no completion marker and restart from
# document one, so whatever is mid-flight at the cap is redone next run -- the completed
# ones still bank permanently.
#
# WORKERS is the thing that actually went wrong before. build_dataset_cache.py's
# --workers defaults to 0 -> os.cpu_count() = 64 on these nodes; note it never imports
# tokenization.DEFAULT_WORKERS (=16), so that constant is dead code and the real default
# is 4x higher than intended. The 2026-08-17 run therefore ran 47 tasks concurrently,
# peaked at MaxRSS 337GB, took 17 oom_kill events (killing ranks 17/45/81/89/94 mid-read)
# and left the survivors thrashing -- shuffles that take ~84 min healthy ran 11h and were
# still unfinished when the 12h limit hit, which is what killed the other 20.
#
# 8 workers is ~7GB/task * 8 = ~56GB against a ~870GB node: no memory pressure at all,
# so the 8 tasks in flight each get the full memory and CPU they need.
#SBATCH --account=infra01
#SBATCH --job-name=build-cache
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --no-requeue
set -euo pipefail

WORKERS="${1:-8}"
REPO=/curriculum-learning-using-attention
VENV=/iopsstor/scratch/cscs/$USER/curriculum-venv

echo "workers: $WORKERS   (64 was the accidental default that OOMed)"

srun --environment="$SLURM_SUBMIT_DIR/clariden.toml" bash -c "
    set -euo pipefail
    source $VENV/bin/activate
    cd $REPO

    # --tasks MUST stay 128. It decides how the corpus is split into shards, so the
    # existing completion markers (and the 103 finished .ds files) only line up with the
    # same value -- changing it silently invalidates every one of them and restarts the
    # whole 151GB build.
    #
    # --splits train validation: train resumes where it left off, and validation has not
    # been built at all yet (build_tokenized_cache does train first, and the job never got
    # past it). Validation is a fixed Salesforce/wikitext wikitext-2-raw-v1 split -- 1
    # file, ~250K tokens -- so it costs minutes, but without it val_ds loads empty and
    # val_ppl silently reports exp(0)=1.0.
    python build_dataset_cache.py \
        --config configs/opus_gpt2xl_muon_fineweb.yaml \
        --tasks 128 \
        --workers $WORKERS \
        --splits train validation
"
