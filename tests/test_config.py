from pathlib import Path

import pytest

from browser_embedding.config import DeploymentConfig, ModelConfig, load_experiment


def test_smoke_config_is_valid_and_within_budget() -> None:
    config, root = load_experiment(Path("configs/smoke.yaml"))
    estimate = config.deployment.assert_fits(config.model)
    assert root == Path.cwd()
    assert estimate.packed_model_bytes < config.deployment.max_model_bytes
    assert estimate.logical_parameters > estimate.trainable_parameters


def test_invalid_head_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=512,
            max_sequence_length=16,
            embedding_dim=32,
            hidden_dim=60,
            num_heads=8,
            num_repeats=2,
            ffn_dim=128,
            output_dim=64,
            matryoshka_dims=(32, 64),
            dropout=0.0,
            padding_idx=0,
        )


def test_budget_violation_fails_before_training() -> None:
    config, _ = load_experiment(Path("configs/smoke.yaml"))
    tiny_budget = DeploymentConfig(
        embedding_format="int4",
        max_model_bytes=100,
        max_bundle_bytes=100,
        max_working_memory_bytes=100,
        wasm_binary_budget_bytes=100,
        tokenizer_asset_budget_bytes=100,
        tokenizer_working_memory_bytes=100,
        enforce_budget=True,
    )
    with pytest.raises(ValueError, match="budget exceeded"):
        tiny_budget.assert_fits(config.model)
