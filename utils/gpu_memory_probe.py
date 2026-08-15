# gpu_memory_probe.py
"""
Measure the peak GPU memory a single experiment config needs by actually
running it for a couple of real training steps (on a tiny slice of the
cached dataset) inside an isolated process, then reporting
torch.cuda.max_memory_allocated().

Run standalone (used by parallel_experiments.py as a subprocess):
    python gpu_memory_probe.py --config <yaml> --dataset-cache <path>

Prints a single parseable line on success:
    PROBE_PEAK_BYTES=<int>

Runs for epochs=2 instead of 1 when cfg.feature_cache_epochs > 0, since the
feature-cache-build code path only triggers starting at epoch 1 (see
build_feature_cache in rl_training.py) and its batch-by-batch forward pass
is the extra GPU memory a real run would pay that a single epoch wouldn't
capture. The cache tensor itself lives on CPU, so a small truncated
dataset is enough to see that path's true GPU peak.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config_from_yaml
from data import get_tokenizer
from models.model import build_model
from models.router import build_router, build_router_for_cfg, get_router_feature_dim
from utils.general_utils import resolve_device
from utils.metrics import MetricsTracker, DiversityTracker
from rl_training import train_router_experiments, train_aux_baseline
from training import train_baseline
from utils.shared_dataset import load_dataset_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-cache", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_memory_probe requires a CUDA device")

    cfg = load_config_from_yaml(args.config)
    probe_cfg = resolve_device(replace(
        cfg,
        use_wandb=False,
        epochs=2 if cfg.feature_cache_epochs > 0 else 1,
    ))

    train_size = max(probe_cfg.pool * 3, 64)
    tokenizer = get_tokenizer(cfg.tokenizer_name)
    train_ds, val_ds = load_dataset_cache(
        args.dataset_cache, probe_cfg, max_train=train_size, max_val=32, warm_cache=False
    )

    # build_model(), NOT TinyGPT directly: memory_signature() already keys the
    # probe cache on (model_type, hf_model_name), so hardcoding TinyGPT here
    # measured a completely different architecture than an
    # 'hf_pretrained' config actually runs -- and understated it badly.
    # __post_init__ copies the checkpoint's hidden size/layer count onto cfg,
    # so the TinyGPT stand-in looked the right *size* while its MLP width came
    # from the YAML's d_ff (2048) instead of the checkpoint's n_inner (6400
    # for GPT2-XL). Activations are the dominant term for the wide-batch
    # baselines, so the scheduler packed runs that could never fit.
    model = build_model(tokenizer.vocab_size, probe_cfg)
    metrics = MetricsTracker(f"probe_{probe_cfg.experiment_name}", use_wandb=False)
    diversity = DiversityTracker(len(train_ds), domain_names=train_ds.domain_names)

    torch.cuda.reset_peak_memory_stats()
    if probe_cfg.run_aux_baseline:
        aux_net = build_router(
            d_input=get_router_feature_dim(probe_cfg, model.block),
            arch="auxnet",
            d_hidden=probe_cfg.aux_net_hidden,
        )
        train_aux_baseline(
            cfg=probe_cfg,
            model=model,
            aux_net=aux_net,
            train_ds=train_ds,
            val_ds=val_ds,
            tokenizer=tokenizer,
            metrics=metrics,
            diversity=diversity,
        )
    elif probe_cfg.run_random_batch_baseline or probe_cfg.run_random_pool_baseline:
        # pool_mult=1 too -- see utils/experiment_worker.py's identical replace()
        # call for why (cfg.pool is a pool_mult*global_batch_size property, so
        # widening global_batch_size alone re-inflates pool by another factor
        # of pool_mult).
        random_cfg = (
            probe_cfg
            if probe_cfg.run_random_batch_baseline
            else replace(probe_cfg, global_batch_size=probe_cfg.pool, pool_mult=1)
        )
        train_baseline(
            cfg=random_cfg,
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            metrics=metrics,
            diversity=diversity,
        )
    else:
        router = build_router_for_cfg(
            probe_cfg,
            sequence_size=model.block,
            vocab_size=tokenizer.vocab_size,
        )
        train_router_experiments(
            cfg=probe_cfg,
            model=model,
            router=router,
            train_ds=train_ds,
            val_ds=val_ds,
            tokenizer=tokenizer,
            metrics=metrics,
            diversity=diversity,
        )
    peak = torch.cuda.max_memory_allocated()
    print(f"PROBE_PEAK_BYTES={peak}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
