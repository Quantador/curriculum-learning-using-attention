"""Sanity checks that the OPUS-comparison configs parse and carry the intended values.
Run directly: python tests/test_opus_configs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_router_config():
    from config import load_config_from_yaml

    cfg = load_config_from_yaml("configs/opus_gpt2xl_muon_fineweb.yaml")
    assert cfg.model_type == "hf_pretrained"
    assert cfg.hf_model_name == "openai-community/gpt2-xl"
    assert cfg.lm_optimizer == "muon"
    assert cfg.lr_muon == 0.01
    assert cfg.lr_lm == 0.002
    assert cfg.grad_clip_norm == 1.0
    assert cfg.max_tokens == 30_000_000_000
    assert cfg.block == 256
    assert cfg.dataset_list == ["HuggingFaceFW/fineweb-100BT"]
    assert cfg.pool_mult == 10
    assert cfg.save_model_at_end is True
    assert cfg.router_architecture == "attention"
    assert cfg.training_algorithm == "ppo"
    assert cfg.reward_signal == "loss_improvement"
    assert cfg.selection_strategy == "topk"
    assert cfg.run_random_batch_baseline is False
    print("OK: router config")


def test_random_baseline_config():
    from config import load_config_from_yaml

    cfg = load_config_from_yaml("configs/opus_gpt2xl_muon_fineweb_random_baseline.yaml")
    assert cfg.run_random_batch_baseline is True
    assert cfg.lm_optimizer == "muon"
    assert cfg.hf_model_name == "openai-community/gpt2-xl"
    assert cfg.max_tokens == 30_000_000_000
    print("OK: random-baseline config")


if __name__ == "__main__":
    test_router_config()
    test_random_baseline_config()
    print("All tests passed.")
