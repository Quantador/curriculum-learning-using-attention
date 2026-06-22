# metrics.py
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

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
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                self.history[k].append(float(v))
        if self.use_wandb:
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
    
    def load(path: str | Path) -> MetricsTracker:
        path = Path(path)
        with path.open("r") as f:
            history = json.load(f)
        tracker = MetricsTracker(name=path.stem)
        tracker.history = {k: v for k, v in history.items()}
        return tracker


class DiversityTracker:
    def __init__(self, dataset_size: int, window: int = 100):
        self.dataset_size = dataset_size
        self.window = window

        self.selection_counts = torch.zeros(dataset_size, dtype=torch.long)
        self.step_selections: List[List[int]] = []
        self.difficulty_counts = {0: 0, 1: 0}

    def update(self, indices: List[int], difficulties) -> None:
        if not indices:
            return

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        self.selection_counts[idx_tensor] += 1

        self.step_selections.append(indices)
        if len(self.step_selections) > self.window:
            self.step_selections.pop(0)

        for d in difficulties:
            if d in self.difficulty_counts:
                self.difficulty_counts[d] += 1

    def get_metrics(self) -> Dict[str, float]:
        counts = self.selection_counts
        selected_mask = counts > 0
        num_selected = selected_mask.sum().item()

        coverage = num_selected / max(1, self.dataset_size)

        if num_selected > 1:
            balance_std = counts[selected_mask].float().std(unbiased=False).item()
        else:
            balance_std = 0.0

        recent = [i for step in self.step_selections for i in step]
        total_recent = len(recent)
        unique_recent = len(set(recent)) if total_recent > 0 else 0
        unique_ratio = unique_recent / max(1, total_recent)

        total_easy = self.difficulty_counts.get(0, 0)
        total_hard = self.difficulty_counts.get(1, 0)
        total_sel = total_easy + total_hard
        if total_sel > 0:
            easy_ratio = total_easy / total_sel
            hard_ratio = total_hard / total_sel
        else:
            easy_ratio = 0.0
            hard_ratio = 0.0

        return {
            "coverage": coverage,
            "balance_std": balance_std,
            "unique_ratio": unique_ratio,
            "easy_ratio": easy_ratio,
            "hard_ratio": hard_ratio,
        }
        
