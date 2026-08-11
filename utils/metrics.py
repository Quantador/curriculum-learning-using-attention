# metrics.py
"""
Metric tracking and logging utilities.

MetricsTracker:
  Accumulates scalar metrics as Python lists in self.history.
  Optionally logs each step to Weights & Biases (if installed and enabled).
  Serialises to / loads from JSON for persistence between training runs.
  Used by every training loop in this project.

DiversityTracker:
  Records which samples (by dataset index) were selected at each step.
  Computes coverage (fraction of dataset ever seen), per-domain selection
  ratio (curriculum direction across however many domains are configured,
  one domain_ratio/{id} key per domain id observed), unique_ratio
  (short-window diversity), and balance_std (selection uniformity across
  the dataset).
  Logged every cfg.log_every steps via MetricsTracker.log().
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None


class MetricsTracker:
    def __init__(self, name: str, use_wandb: bool = False):
        self.name = name
        self.use_wandb = use_wandb and (wandb is not None)
        self.history: Dict[str, List[float]] = defaultdict(list)

    def log(self, **kwargs: Any) -> None:
        """Log scalar metrics to history and optionally to W&B.

        Non-numeric values are silently skipped (e.g. epoch strings).
        Skips the W&B call once wandb.run is None -- e.g. after the training
        loop's own wandb.finish() has already run but a caller still logs a
        post-hoc summary metric (total_time_s) into this same tracker.
        """
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                self.history[k].append(float(v))
        if self.use_wandb and wandb.run is not None:
            wandb.log(kwargs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.history, f)

    def get_final_ppl(self) -> Optional[float]:
        vals = self.history.get("val_ppl")
        if not vals:
            return None
        return vals[-1]

    def get_total_time(self) -> Optional[float]:
        """Total wall-clock seconds for the run (logged once, see
        utils/experiment_worker.py's total_time_s)."""
        vals = self.history.get("total_time_s")
        if not vals:
            return None
        return vals[-1]
    
    def load(path: str | Path) -> MetricsTracker:
        """
        Load a MetricsTracker from a JSON file previously saved by .save().

        Returns an empty tracker if the file doesn't exist yet (e.g. no
        compare.py baseline/router run has been done in this save_dir) --
        callers only use this for optional comparison printouts, which
        already handle missing val_ppl history gracefully.
        """
        path = Path(path)
        tracker = MetricsTracker(name=path.stem)
        if not path.exists():
            return tracker
        with path.open("r") as f:
            history = json.load(f)
        tracker.history = {k: v for k, v in history.items()}
        return tracker


class DiversityTracker:
    def __init__(self, dataset_size: int, domain_names: Optional[List[str]] = None, window: int = 100):
        self.dataset_size = dataset_size
        self.window = window
        # domain_id -> name (e.g. train_ds.domain_names), used to label
        # domain_ratio/{name} metrics with real names instead of bare ids.
        # None falls back to the id itself (str(domain_id)).
        self.domain_names = domain_names

        self.selection_counts = torch.zeros(dataset_size, dtype=torch.long)
        self.step_selections: List[List[int]] = []
        self.domain_counts = {}

    def _domain_label(self, domain_id: int) -> str:
        if self.domain_names is not None and 0 <= domain_id < len(self.domain_names):
            return self.domain_names[domain_id]
        return str(domain_id)

    def update(self, indices: List[int], domains) -> None:
        if not indices:
            return

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        self.selection_counts[idx_tensor] += 1

        self.step_selections.append(indices)
        if len(self.step_selections) > self.window:
            self.step_selections.pop(0)

        for d in domains:
            if d not in self.domain_counts:
                self.domain_counts[d] = 0
            self.domain_counts[d] += 1

    def get_metrics(self, world_size: int = 1) -> Dict[str, float]:
        """
        Return a dict of diversity metrics accumulated since instantiation.

        Keys:
          coverage      — fraction of dataset samples selected at least once [0,1]
          balance_std   — std of selection counts (lower = more uniform coverage)
          unique_ratio  — fraction of unique samples in the last `window` steps [0,1]
          domain_ratio/{name} — fraction of selected samples from that domain
                              [0,1], one key per distinct domain observed so
                              far. Sharing the "domain_ratio/" prefix across
                              domains groups them into one section/panel in
                              the W&B UI automatically.

        world_size > 1: each rank only ever calls update() with its own
        DistributedSampler-sharded slice of the dataset (see data.py), so
        self.selection_counts/domain_counts/step_selections are each only
        that rank's own partial view. Pass cfg.world_size here to merge
        every rank's view before computing the stats above -- selection_counts
        via all_reduce (dataset-index tensor, cheap to sum elementwise),
        domain_counts/step_selections via all_gather_object (small Python
        objects, not tensors). Every rank must call this the same number of
        times with the same world_size (it's a collective) -- true here
        since it's only ever invoked from a cfg.log_every-gated block that
        every rank reaches in lockstep. Uses local copies throughout, so
        self.* state is never mutated by this call and repeated calls don't
        double-count already-merged contributions.
        """
        counts = self.selection_counts
        domain_counts = self.domain_counts
        step_selections = self.step_selections

        if world_size > 1:
            counts = counts.clone()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

            gathered_domain_counts: List[Optional[dict]] = [None] * world_size
            dist.all_gather_object(gathered_domain_counts, self.domain_counts)
            merged_domain_counts: Dict[int, int] = defaultdict(int)
            for dc in gathered_domain_counts:
                for domain_id, count in dc.items():
                    merged_domain_counts[domain_id] += count
            domain_counts = merged_domain_counts

            gathered_step_selections: List[Optional[List[List[int]]]] = [None] * world_size
            dist.all_gather_object(gathered_step_selections, self.step_selections)
            step_selections = [
                step for rank_steps in gathered_step_selections for step in rank_steps
            ]

        selected_mask = counts > 0
        num_selected = selected_mask.sum().item()

        coverage = num_selected / max(1, self.dataset_size)

        if num_selected > 1:
            balance_std = counts[selected_mask].float().std(unbiased=False).item()
        else:
            balance_std = 0.0

        recent = [i for step in step_selections for i in step]
        total_recent = len(recent)
        unique_recent = len(set(recent)) if total_recent > 0 else 0
        unique_ratio = unique_recent / max(1, total_recent)

        total_sel = sum(domain_counts.values())
        domain_ratios = {
            f"domain_ratio/{self._domain_label(domain_id)}": (count / total_sel if total_sel > 0 else 0.0)
            for domain_id, count in sorted(domain_counts.items())
        }

        return {
            "coverage": coverage,
            "balance_std": balance_std,
            "unique_ratio": unique_ratio,
            **domain_ratios,
        }
        
