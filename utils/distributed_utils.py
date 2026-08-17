# distributed_utils.py
"""
PyTorch process-group lifecycle and model-parallelization for
parallel_experiments.py.

setup_distributed() / cleanup_distributed() wrap torch.distributed's process
group lifecycle, falling back to a single-process no-op when launched
without torchrun so every other entry point is unaffected.

wrap_model() / wrap_replica() / eval_handles() are the single place that
knows about cfg.distributed ('DDP' vs 'FSDP'). All three training loops
(training.train_baseline, rl_training.train_router_experiments,
rl_training.train_aux_baseline) go through them rather than each
hand-rolling `DDP(model, ...)` plus its own `model.module if isinstance(...)`
unwrapping -- those hand-rolled forms silently did the wrong thing the moment
a second parallelization strategy existed.

Unlike the older DDP prototype this replaces, there is no dataset-chunk
broadcasting here: utils.shared_dataset.load_dataset_cache() already reads a
pre-tokenized, mmap-backed on-disk cache, which every rank can load
independently and identically -- no rank-0-builds-then-broadcasts dance
needed.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

# FSDP2's public home is torch.distributed.fsdp only from torch 2.6 on; in
# 2.5 (what the current image ships) the same objects live under the private
# _composable path. Import failure is NOT fatal here on purpose: FSDP is one
# optional cfg.distributed value, and a module-level ImportError for it would
# take down every training entry point that merely imports this file --
# including all the single-GPU and DDP runs that never touch FSDP at all.
try:  # torch >= 2.6
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    _FSDP_IMPORT_ERROR = None
except ImportError:
    try:  # torch 2.5
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
        _FSDP_IMPORT_ERROR = None
    except ImportError as exc:
        fully_shard = MixedPrecisionPolicy = None
        _FSDP_IMPORT_ERROR = exc

_MP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize the process group from torchrun's env vars.

    Returns (rank, local_rank, world_size). If WORLD_SIZE is not set (i.e.
    not launched via torchrun), returns (0, 0, 1) without touching
    torch.distributed at all -- the single-process case.
    """
    if "WORLD_SIZE" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        # No GPUs visible (e.g. a local dry run, or this module's own CPU
        # smoke tests) -- gloo lets the same control flow be exercised
        # without real GPU hardware.
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _ddp(module: nn.Module, cfg) -> nn.Module:
    # device_ids must be None for CPU modules (only single/multi-GPU modules
    # accept it) -- only relevant for the gloo/CPU smoke-test path, since real
    # DDP training always runs on CUDA.
    return DDP(module, device_ids=[cfg.local_rank] if torch.cuda.is_available() else None)


def wrap_model(model: nn.Module, cfg) -> nn.Module:
    """Parallelize the LM across ranks per cfg.distributed. No-op at world_size 1.

    'DDP' replicates; 'FSDP' shards with FSDP2's fully_shard. The FSDP path
    shards each transformer block first and then the root module, which is
    the documented FSDP2 pattern: per-block sharding is what lets the
    all-gather for block N+1 overlap with block N's compute, and the root
    call is what covers the embeddings / lm_head left outside the blocks.

    Note FSDP2 mutates in place and returns the SAME object -- there is no
    wrapper and no `.module`. That is precisely why eval_handles() exists.
    """
    if cfg.world_size <= 1:
        return model

    if cfg.distributed == "DDP":
        return _ddp(model, cfg)

    if cfg.distributed == "FSDP":
        if fully_shard is None:
            raise RuntimeError(
                "cfg.distributed='FSDP' needs FSDP2 (fully_shard), which this "
                f"torch ({torch.__version__}) does not provide under either "
                f"torch.distributed.fsdp or torch.distributed._composable.fsdp: "
                f"{_FSDP_IMPORT_ERROR}"
            )
        mp_dtype = _MP_DTYPES[cfg.fsdp_mixed_precision]
        kwargs = {}
        if mp_dtype is not None:
            # reduce_dtype stays fp32: gradient reduction is where low-precision
            # accumulation actually costs you accuracy, and it is cheap to keep
            # wide relative to the param all-gathers.
            kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=mp_dtype, reduce_dtype=torch.float32
            )
        for block in transformer_blocks(model):
            fully_shard(block, **kwargs)
        fully_shard(model, **kwargs)
        return model

    raise NotImplementedError(f"There is no distributed training with {cfg.distributed!r}")


def wrap_replica(module: nn.Module, cfg) -> nn.Module:
    """Parallelize a SMALL module (router, aux net) -- always by replication.

    Sharding these would be a pessimization, not a saving: the routers are a
    few thousand parameters (LinearRouter is a single Linear(d_input, 1)), so
    FSDP2 would add an all-gather and a reduce-scatter per step to distribute
    a tensor that already fits everywhere. Replicate them even when the LM
    itself is sharded.
    """
    return _ddp(module, cfg) if cfg.world_size > 1 else module


def transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """The model's repeated transformer blocks, for per-block FSDP sharding.

    Delegates to the model's own accessor (TinyGPT and HFCausalLM keep their
    blocks in different places -- `self.tr.layers` vs the HF backbone's `.h`
    or `.layers`) rather than reaching into either layout from here.
    """
    accessor = getattr(model, "transformer_blocks", None)
    if accessor is None:
        raise AttributeError(
            f"{type(model).__name__} does not expose transformer_blocks(); add one "
            f"(see TinyGPT/HFCausalLM in models/model.py) before sharding it with FSDP."
        )
    return accessor()


def eval_handles(model: nn.Module, cfg) -> Tuple[nn.Module, bool]:
    """Returns (module to call evaluate() on, whether THIS rank must call it).

    The two strategies need opposite answers, which is the subtle part:

    DDP  -> (unwrapped module, rank == 0). DDP's forward() broadcasts module
            buffers whenever the last grad-enabled forward left
            require_forward_param_sync set (true for HF models with
            registered buffers, e.g. GPT-2's attn.bias) -- a collective the
            other ranks aren't there to join. Evaluating `.module` sidesteps
            DDP's forward entirely, so rank 0 can evaluate alone.

    FSDP -> (the sharded module, every rank). There is no unwrapped module to
            escape to: no rank holds a complete copy of the parameters, so
            the forward MUST all-gather and every rank must participate or
            NCCL hangs until timeout. val_ds is identical and evaluate() is
            deterministic, so all ranks compute the same number and callers
            can keep logging only rank 0's copy without a reduction.
    """
    if cfg.world_size > 1 and cfg.distributed == "FSDP":
        return model, True
    return (model.module if isinstance(model, DDP) else model), cfg.rank == 0


def full_state_dict(module: nn.Module, cfg) -> dict:
    """Gather module's full, unsharded state dict for checkpointing.

    Branches on how THIS module is actually wrapped, not on cfg.distributed:
    wrap_replica() always DDP-wraps (routers/aux nets are replicated even
    when cfg.distributed == 'FSDP' shards the LM, see its docstring), so a
    router passed in under an FSDP run must still take the DDP branch below,
    not the FSDP one.

    DDP/unwrapped: every rank already holds an identical full replica, so
    this just reads it locally off `.module` -- no collective needed.

    FSDP (mutates in place, no wrapper -- see wrap_model()'s docstring):
    gathering the sharded DTensor params is a collective all-gather, so this
    branch must be called on EVERY rank; the assembled dict then only lands
    on rank 0 (get_model_state_dict's documented behavior for
    full_state_dict=True) -- other ranks' return value is not meant to be
    used.
    """
    if isinstance(module, DDP):
        return {k: v.detach().cpu() for k, v in module.module.state_dict().items()}
    if cfg.world_size > 1 and cfg.distributed == "FSDP":
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, get_model_state_dict,
        )
        return get_model_state_dict(
            module, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}
