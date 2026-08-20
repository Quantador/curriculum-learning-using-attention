"""Tests for utils/muon_optimizer.py. Run directly: python tests/test_muon_optimizer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# Monkey-patch torch.library for torchvision compatibility (torch 2.2.0 doesn't have register_fake)
if not hasattr(torch.library, 'register_fake'):
    class _FakeLibraryCompat:
        @staticmethod
        def register_fake(name):
            def decorator(func):
                return func
            return decorator

    torch.library.register_fake = _FakeLibraryCompat.register_fake


def test_muon_reduces_loss_on_toy_linear_regression():
    from utils.muon_optimizer import Muon

    torch.manual_seed(0)
    target = torch.randn(8, 4)
    layer = torch.nn.Linear(4, 8, bias=False)
    x = torch.randn(32, 4)
    y = x @ target.T

    opt = Muon(layer.parameters(), lr=0.05, momentum=0.9, momentum_warmup_steps=1, ns_steps=5)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        pred = layer(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1, (
        f"Muon did not converge on toy linear regression: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    print(f"OK: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_gpt2_xl_partition_matches_opus_table1():
    """OPUS Table 1: only in-block attn/mlp 2D weights go to Muon; embeddings, LM head, and
    every norm/bias go to AdamW. Uses torch.device("meta") so this only fetches GPT-2 XL's
    small config.json (network access, no weight download) and never materializes real
    parameter storage -- fast even though the real model is 1.5B params."""
    from config import Config
    from models.model import build_model
    from utils.muon_optimizer import build_muon_param_groups

    cfg = Config(model_type="hf_pretrained", hf_model_name="openai-community/gpt2-xl")
    with torch.device("meta"):
        model = build_model(vocab_size=50257, cfg=cfg)

    id_to_name = {id(p): n for n, p in model.named_parameters()}
    muon_groups, adamw_params = build_muon_param_groups(model)

    muon_names = sorted(id_to_name[id(p)] for g in muon_groups for p in g["params"])
    adamw_names = sorted(id_to_name[id(p)] for p in adamw_params)

    for name in muon_names:
        assert "transformer.h." in name, f"unexpected Muon param: {name}"
        assert name.endswith((
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight", "mlp.c_proj.weight",
        )), f"unexpected Muon param: {name}"

    assert any(n.endswith("transformer.wte.weight") for n in adamw_names), f"wte.weight not found in {adamw_names[:5]}"
    assert any(n.endswith("ln_1.weight") for n in adamw_names)
    assert any(n.endswith("attn.c_attn.bias") for n in adamw_names)

    # GPT-2's Conv1D stores weight as (in, out) -- verify the recorded orientation is
    # correctly un-transposed (out_features/in_features), not the raw Conv1D shape.
    for g in muon_groups:
        name = id_to_name[id(g["params"][0])]
        if name.endswith("mlp.c_fc.weight"):
            assert g["in_features"] == 1600 and g["out_features"] == 6400, (name, g)
        if name.endswith("mlp.c_proj.weight"):
            assert g["in_features"] == 6400 and g["out_features"] == 1600, (name, g)

    n_layers = 48  # GPT-2 XL
    assert len(muon_groups) == n_layers * 4, f"expected {n_layers * 4} Muon groups, got {len(muon_groups)}"
    print(f"OK: {len(muon_groups)} Muon param groups, {len(adamw_params)} AdamW params")


def test_tinygpt_partition_includes_in_proj_weight():
    """Verify TinyGPT's fused attention weights (in_proj_weight from nn.MultiheadAttention)
    are correctly routed to Muon, not silently dropped to AdamW. TinyGPT is the project's
    default model_type, so this path must work correctly."""
    from config import Config
    from models.model import build_model
    from utils.muon_optimizer import build_muon_param_groups

    cfg = Config(model_type="tiny_gpt", d_model=256, n_layers=2, n_heads=4)
    with torch.device("meta"):
        model = build_model(vocab_size=256, cfg=cfg)

    id_to_name = {id(p): n for n, p in model.named_parameters()}
    muon_groups, adamw_params = build_muon_param_groups(model)

    muon_names = sorted(id_to_name[id(p)] for g in muon_groups for p in g["params"])
    adamw_names = sorted(id_to_name[id(p)] for p in adamw_params)

    # TinyGPT should have in_proj_weight (fused QKV) from nn.MultiheadAttention in Muon,
    # not silently dropped to AdamW
    in_proj_found = any("in_proj_weight" in name for name in muon_names)
    assert in_proj_found, f"in_proj_weight not found in Muon groups. Muon: {muon_names[:5]}, AdamW: {adamw_names[:5]}"

    # Verify at least one in_proj_weight is in Muon and has correct dimensions
    for g in muon_groups:
        name = id_to_name[id(g["params"][0])]
        if "in_proj_weight" in name:
            # nn.MultiheadAttention.in_proj_weight shape: (3*embed_dim, embed_dim)
            # So out_features should be 3*d_model and in_features should be d_model
            assert g["out_features"] == 3 * cfg.d_model, (
                f"in_proj_weight out_features mismatch: {g['out_features']} "
                f"!= 3*{cfg.d_model}"
            )
            assert g["in_features"] == cfg.d_model, (
                f"in_proj_weight in_features mismatch: {g['in_features']} != {cfg.d_model}"
            )
            break
    else:
        raise AssertionError("No in_proj_weight found in Muon groups during dimension check")

    print(f"OK: {len(muon_groups)} Muon param groups (includes in_proj_weight), "
          f"{len(adamw_params)} AdamW params")


def _make_tiny_hf_gpt2():
    """A tiny (not XL) HF GPT-2 model, built the same minimal way in every test below that
    needs a real HFCausalLM without paying __init__'s config-fetch/download cost."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from models.model import HFCausalLM

    hf_config = AutoConfig.from_pretrained(
        "openai-community/gpt2", n_layer=2, n_head=2, n_embd=32, vocab_size=100,
    )
    model = HFCausalLM.__new__(HFCausalLM)  # bypass __init__'s config fetch; set up minimally
    torch.nn.Module.__init__(model)
    model.hf = AutoModelForCausalLM.from_config(hf_config)
    model.vocab_size = 100
    model.block = 16
    model.d_model = 32
    model._pos_embed_module = model.hf.transformer.wpe
    model.supports_embedder_mode = True
    return model


def test_build_optimizer_muon_path_trains_a_tiny_gpt2():
    """End-to-end sanity check with the real factory + a tiny (not XL) HF GPT-2 config, so
    this runs fast: build_optimizer(cfg.lm_optimizer='muon') should train without error and
    reduce loss over a few steps, exercising both the Muon and AdamW sub-groups together."""
    from config import Config
    from utils.muon_optimizer import build_optimizer

    model = _make_tiny_hf_gpt2()
    cfg = Config(lm_optimizer="muon", lr_muon=0.02, lr_lm=0.001, grad_clip_norm=1.0)
    opt = build_optimizer(model, cfg)

    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4, 16))
    losses = []
    for _ in range(20):
        opt.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 100), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"OK: tiny GPT-2 + Muon loss {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_build_muon_param_groups_handles_ddp_wrapped_model():
    """build_muon_param_groups()'s DDP-unwrap branch (`model.module if isinstance(model, DDP)
    else model`) is only exercised in real training runs when a model is wrapped via
    utils.distributed_utils.wrap_model() *before* build_optimizer() is called at all three
    training-loop call sites. No existing test constructed a real DDP-wrapped model, so a
    subtly wrong unwrap could only ever surface during an expensive multi-GPU run.

    This spins up a real, single-process CPU process group (gloo backend) so
    torch.nn.parallel.DistributedDataParallel can legitimately wrap a model, then checks that
    build_muon_param_groups()/build_optimizer() partition the DDP-wrapped model's parameters
    into the exact same Muon/AdamW buckets (by parameter identity, not just count) as the
    unwrapped model -- i.e. wrapping in DDP must not change which parameters land where.
    """
    import os
    import socket
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from config import Config
    from utils.muon_optimizer import build_muon_param_groups, build_optimizer, MultiOptimizer

    # This torch build's gloo/TCPStore rendezvous on Windows fails ("use_libuv was requested
    # but PyTorch was built without libuv support") unless libuv is explicitly disabled.
    os.environ.setdefault("USE_LIBUV", "0")

    model = _make_tiny_hf_gpt2()
    muon_groups_ref, adamw_ref = build_muon_param_groups(model)
    muon_ids_ref = {id(p) for g in muon_groups_ref for p in g["params"]}
    adamw_ids_ref = {id(p) for p in adamw_ref}

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0,
    )
    try:
        wrapped = DDP(model, device_ids=None)  # CPU-only DDP: no device_ids
        assert isinstance(wrapped, DDP)

        muon_groups_ddp, adamw_ddp = build_muon_param_groups(wrapped)
        muon_ids_ddp = {id(p) for g in muon_groups_ddp for p in g["params"]}
        adamw_ids_ddp = {id(p) for p in adamw_ddp}
        assert muon_ids_ddp == muon_ids_ref, "DDP wrap changed which params land in Muon"
        assert adamw_ids_ddp == adamw_ids_ref, "DDP wrap changed which params land in AdamW"

        # Also exercise it through the real factory, exactly as the training loops call it
        # (build_optimizer(model, cfg) on the already wrap_model()-wrapped model).
        cfg = Config(lm_optimizer="muon", lr_muon=0.02, lr_lm=0.001)
        opt = build_optimizer(wrapped, cfg)
        assert isinstance(opt, MultiOptimizer)
    finally:
        dist.destroy_process_group()

    print(
        f"OK: DDP-wrapped partition matches unwrapped "
        f"({len(muon_ids_ddp)} Muon, {len(adamw_ids_ddp)} AdamW)"
    )


def test_muon_with_fsdp_is_rejected():
    """Muon under FSDP2 is silently WRONG, not merely unsupported: fully_shard leaves params
    and grads as dim-0-sharded DTensors, and Newton-Schulz orthogonalization of a row-shard
    (X @ X.T over part of the rows) is not the corresponding shard of the full matrix's
    orthogonalization -- so the updates would be quietly incorrect. _save_checkpoint() also
    has no FSDP2 full-state-dict path. Config must refuse the combination up front rather
    than let a GPT-2 XL-sized run (exactly the size someone reaches for FSDP at) produce
    garbage over several days.
    """
    from config import Config, ExperimentConfig

    for cls in (Config, ExperimentConfig):
        try:
            cls(lm_optimizer="muon", distributed="FSDP")
        except ValueError as exc:
            assert "muon" in str(exc) and "FSDP" in str(exc), exc
        else:
            raise AssertionError(f"{cls.__name__}(lm_optimizer='muon', distributed='FSDP') "
                                 f"did not raise ValueError")

    # The combinations that ARE supported must keep working.
    Config(lm_optimizer="muon", distributed="DDP")
    Config(lm_optimizer="adamw", distributed="FSDP")
    print("OK: lm_optimizer='muon' + distributed='FSDP' is rejected; DDP/adamw combos still build")


if __name__ == "__main__":
    test_muon_reduces_loss_on_toy_linear_regression()
    test_gpt2_xl_partition_matches_opus_table1()
    test_tinygpt_partition_includes_in_proj_weight()
    test_build_optimizer_muon_path_trains_a_tiny_gpt2()
    test_build_muon_param_groups_handles_ddp_wrapped_model()
    test_muon_with_fsdp_is_rejected()
    print("All tests passed.")
