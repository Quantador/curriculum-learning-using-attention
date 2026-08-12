import torch
import re
import os
import sys
import subprocess
import yaml
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple
from consts import EXPERIMENT_PROFILES, PER_PROC_BUFFER, CONTEXT_OVERHEAD_BYTES, EXPERIMENTAL_FIELDS
from config import ExperimentConfig
from dataclasses import asdict

def get_baseline_config() -> Dict[str, Any]:
    """Get the baseline values for all experimental fields."""
    return {field: values[0] for field, values in EXPERIMENTAL_FIELDS.items()}

def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_profile_fields(profile: str | None) -> Dict[str, tuple[Any, List[Any]]] | None:
    if not profile:
        return None
    if profile in EXPERIMENT_PROFILES:
        return EXPERIMENT_PROFILES[profile]
    normalized = profile.replace("-", "_").replace(" ", "_")
    if normalized in EXPERIMENT_PROFILES:
        return EXPERIMENT_PROFILES[normalized]
    raise ValueError(
        f"Unknown profile '{profile}'. Available: {', '.join(sorted(set(EXPERIMENT_PROFILES.keys())))}"
    )

def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(asdict(cfg), f)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def tee_stdio(log_path: Path):
    """Mirror stdout/stderr to `log_path` for the duration of the block, in
    addition to the real console -- unlike run_scheduler()'s subprocess path
    (stdout=log_f fully replaces the console), run_ddp_sweep() runs each
    experiment in-process under torchrun, where an external supervisor
    (Slurm/k8s/etc.) may already be the only thing capturing stdout, so we
    tee rather than redirect to avoid losing that visibility."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        tee_out, tee_err = _Tee(sys.stdout, f), _Tee(sys.stderr, f)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_out, tee_err
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

def memory_signature(cfg: ExperimentConfig) -> Tuple[Any, ...]:
    """Fields that plausibly change GPU memory use. Configs sharing a
    signature are assumed to need the same amount of GPU memory, so we only
    probe once per signature instead of once per config.

    Uses the *effective* LM batch size, not the raw cfg.batch: run_random_pool_baseline
    widens the selected batch to cfg.pool at runtime (see utils/experiment_worker.py),
    so probing it under cfg.batch would understate its real peak memory and risk an
    OOM once the scheduler packs it alongside other jobs."""
    effective_batch = cfg.pool if cfg.run_random_pool_baseline else cfg.batch
    return (
        cfg.model_type, cfg.hf_model_name,
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.n_chunks,
        effective_batch, cfg.block, cfg.pool_mult,
        cfg.training_algorithm, cfg.ppo_epochs, cfg.grpo_group_size,
        cfg.router_architecture, cfg.router_n_heads,
        cfg.enable_text_hierarchical, cfg.hierarchical_representation, cfg.hierarchical_layer_index,
        cfg.feature_cache_epochs > 0, cfg.feature_cache_batch_size,
    )

def query_free_memory_bytes(gpu_index: int) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    return int(out.decode().strip()) * 1024 * 1024

def probe_signature(cfg: ExperimentConfig, dataset_cache: Path, scratch_dir: Path, gpu_index: int) -> int:
    """dataset_cache is the chunks.pt path matching cfg's own dataset signature."""
    cfg_path = scratch_dir / "probe_configs" / f"{safe_name(cfg.experiment_name)}.yaml"
    dump_config(cfg, cfg_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    proc = subprocess.run(
        [
            sys.executable, "utils/gpu_memory_probe.py",
            "--config", str(cfg_path),
            "--dataset-cache", str(dataset_cache),
        ],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(
            f"GPU memory probe failed for a config like '{cfg.experiment_name}' "
            f"(exit {proc.returncode}). See output above."
        )

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("PROBE_PEAK_BYTES="):
            return int(line.split("=", 1)[1])

    raise RuntimeError(
        f"GPU memory probe for '{cfg.experiment_name}' did not report PROBE_PEAK_BYTES. "
        f"stdout:\n{proc.stdout}"
    )

def compute_costs(
    configs: List[ExperimentConfig], dataset_cache_by_name: Dict[str, Path], scratch_dir: Path, gpu_index: int
) -> Dict[str, int]:
    """Returns {experiment_name: cost_bytes}, probing once per distinct memory signature.

    dataset_cache_by_name maps each config's experiment_name to the tokenized
    cache directory matching that config's own dataset signature (see
    utils/shared_dataset.require_dataset_cache).
    """
    signature_of = {cfg.experiment_name: memory_signature(cfg) for cfg in configs}
    representatives: Dict[Tuple[Any, ...], ExperimentConfig] = {}
    for cfg in configs:
        representatives.setdefault(signature_of[cfg.experiment_name], cfg)

    print(f"\n=== Probing GPU memory for {len(representatives)} distinct config signature(s) ===")
    peak_by_signature: Dict[Tuple[Any, ...], int] = {}
    for i, (sig, rep_cfg) in enumerate(representatives.items(), 1):
        peak = probe_signature(rep_cfg, dataset_cache_by_name[rep_cfg.experiment_name], scratch_dir, gpu_index)
        peak_by_signature[sig] = peak
        print(f"  [{i}/{len(representatives)}] like '{rep_cfg.experiment_name}': {peak / 1e9:.2f} GB peak")

    return {
        cfg.experiment_name: int(peak_by_signature[signature_of[cfg.experiment_name]] * PER_PROC_BUFFER)
        + CONTEXT_OVERHEAD_BYTES
        for cfg in configs
    }