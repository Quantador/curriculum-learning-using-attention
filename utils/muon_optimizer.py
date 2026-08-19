"""Muon optimizer and OPUS-style hybrid Muon+AdamW parameter partitioning.

Implements the optimizer OPUS (arXiv:2602.05400) uses for its GPT-2 pretraining results:
momentum + Newton-Schulz orthogonalization on 2D matrix parameters inside transformer blocks,
with AdamW handling everything else (embeddings, LM head, norms, biases). See OPUS's
Section 4.2/Table 1 for the source formulas.

Distributed support: DDP ONLY. No distributed-specific changes are needed for DDP -- its
Reducer all-reduces `.grad` during backward(), so by the time step() runs each rank holds the
complete, correct gradient matrix that Newton-Schulz needs, exactly the contract AdamW already
relies on. That reasoning does NOT carry over to FSDP2: `fully_shard` leaves parameters and
gradients as dim-0-sharded DTensors, and orthogonalizing a row-shard (X @ X.T over part of the
rows) is not the corresponding shard of the full matrix's orthogonalization -- it would
silently produce wrong updates (or an unexpected implicit all-gather, depending on DTensor op
coverage). utils/experiment_worker.py's _save_checkpoint() likewise has a DDP `.module` unwrap
but no FSDP2 full-state-dict path, so an FSDP checkpoint would not reload. Config.__post_init__
therefore rejects lm_optimizer='muon' together with distributed='FSDP' outright; both gaps must
be closed before that restriction is lifted. (The design spec's "Distributed interaction"
section predates this finding and overstates FSDP2 compatibility.)
"""
from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate zeroth-power (orthogonalizing) transform of G via a quintic Newton-Schulz
    iteration, run in bf16 for speed -- this is Muon's replacement for an exact SVD-based
    UV^T orthogonalization. Coefficients (a, b, c) are the standard published Muon values.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.pytorch_utils import Conv1D


def _param_orientation(module: torch.nn.Module, param_name: str) -> tuple[int, int] | None:
    """(out_features, in_features) for a 2D weight, or None for anything else (biases,
    embeddings, norms). Handles HF's Conv1D, whose weight shape is (in, out) -- the OPPOSITE
    of nn.Linear's (out, in). Also recognizes nn.MultiheadAttention.in_proj_weight (fused QKV),
    which follows the normal (out, in) convention like nn.Linear."""
    # Handle fused attention projection (TinyGPT's nn.MultiheadAttention.in_proj_weight)
    if param_name == "in_proj_weight":
        if isinstance(module, torch.nn.MultiheadAttention):
            out_features, in_features = module.in_proj_weight.shape
            return out_features, in_features
        return None

    if param_name != "weight":
        return None
    if isinstance(module, Conv1D):
        in_features, out_features = module.weight.shape
        return out_features, in_features
    if isinstance(module, torch.nn.Linear):
        out_features, in_features = module.weight.shape
        return out_features, in_features
    return None


def build_muon_param_groups(model: torch.nn.Module) -> tuple[list[dict], list[torch.nn.Parameter]]:
    """Partition model's parameters per OPUS Table 1: 2D weights inside
    model.transformer_blocks() go to Muon (with the correct out/in orientation recorded for
    the effective-LR rescale); everything else goes to a flat AdamW list.

    Accepts model wrapped in DistributedDataParallel (unwraps via .module to find
    transformer_blocks() -- DDP only proxies registered submodules, not custom methods, same
    reasoning as extract_hierarchical_hidden() in models/model.py).

    DDP only. FSDP2-wrapped models would need no *unwrapping* (fully_shard mutates in place,
    same object, same transformer_blocks()), but Muon itself is NOT correct under FSDP2 -- see
    the module docstring; Config.__post_init__ rejects that combination outright.
    """
    m = model.module if isinstance(model, DDP) else model

    block_modules: set[torch.nn.Module] = set()
    for block in m.transformer_blocks():
        block_modules.update(block.modules())

    muon_groups: list[dict] = []
    seen_ids: set[int] = set()
    for module in block_modules:
        for name, param in module.named_parameters(recurse=False):
            if id(param) in seen_ids:
                continue
            orientation = _param_orientation(module, name)
            if orientation is not None:
                seen_ids.add(id(param))
                out_features, in_features = orientation
                muon_groups.append({
                    "params": [param],
                    "out_features": out_features,
                    "in_features": in_features,
                })

    adamw_params: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        adamw_params.append(param)

    return muon_groups, adamw_params


class Muon(torch.optim.Optimizer):
    """Momentum + Newton-Schulz orthogonalization + shape-aware effective-LR rescale.

    Reads optional per-group `out_features`/`in_features` (set by build_muon_param_groups()
    below) for the effective-LR rescale eta_eff = lr * sqrt(max(1, out/in)) -- this matters
    because HF's GPT-2 Conv1D layers store weight as (in, out), the OPPOSITE of nn.Linear's
    (out, in), so raw p.shape[0]/p.shape[1] is wrong for those layers. Falls back to raw
    p.shape when out_features/in_features aren't set (e.g. a plain nn.Linear passed directly).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 momentum_warmup_steps: int = 300, ns_steps: int = 5):
        defaults = dict(
            lr=lr, momentum=momentum, momentum_warmup_steps=momentum_warmup_steps,
            ns_steps=ns_steps, out_features=None, in_features=None,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum_final = group["momentum"]
            warmup_steps = group["momentum_warmup_steps"]
            ns_steps = group["ns_steps"]
            out_features = group["out_features"]
            in_features = group["in_features"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["step"] = 0
                state["step"] += 1
                # 0.85 -> momentum_final over the first warmup_steps steps (OPUS Sec 6.1).
                momentum = 0.85 + (momentum_final - 0.85) * min(1.0, state["step"] / warmup_steps)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=1 - momentum)
                # Nesterov-style "double-smoothed" direction (OPUS Eq. 3):
                # q_{t+1} = (1-momentum)*g_t + momentum*m_{t+1}
                g_nesterov = g.mul(1 - momentum).add_(buf, alpha=momentum)

                update = zeropower_via_newtonschulz5(g_nesterov, steps=ns_steps)

                rows = out_features if out_features is not None else p.shape[0]
                cols = in_features if in_features is not None else p.shape[1]
                eff_lr = lr * max(1.0, rows / cols) ** 0.5
                p.add_(update.to(p.dtype), alpha=-eff_lr)
        return loss


class MultiOptimizer:
    """Wraps several torch.optim.Optimizer instances behind the single zero_grad()/step()
    interface every opt_lm call site in training.py/rl_training.py already uses, so those
    call sites don't need to know whether cfg.lm_optimizer split the model's parameters
    across more than one underlying optimizer."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer | MultiOptimizer:
    """Factory for the LM optimizer, replacing the hardcoded
    torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm, weight_decay=0.0) call sites in
    training.py / rl_training.py. cfg.lm_optimizer='adamw' (default) is byte-identical to that
    old call; 'muon' builds OPUS's hybrid per build_muon_param_groups() above, with the AdamW
    sub-group using OPUS's stated betas for the hybrid setup (beta1=0.8, beta2=0.95, eps=1e-8).
    """
    if cfg.lm_optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm, weight_decay=0.0)

    muon_groups, adamw_params = build_muon_param_groups(model)
    muon_opt = Muon(muon_groups, lr=cfg.lr_muon, momentum=0.95, momentum_warmup_steps=300, ns_steps=5)
    adamw_opt = torch.optim.AdamW(
        adamw_params, lr=cfg.lr_lm, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.0,
    )
    return MultiOptimizer([muon_opt, adamw_opt])
