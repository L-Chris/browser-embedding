from pathlib import Path

import torch

from browser_embedding.config import load_experiment
from browser_embedding.model import BrowserEncoder
from browser_embedding.quantization import set_quantization_strength, ternary_health


def test_model_forward_and_shared_depth() -> None:
    config, _ = load_experiment(Path("configs/smoke.yaml"))
    model = BrowserEncoder(config.model).eval()
    input_ids = torch.randint(1, config.model.vocab_size, (3, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -3:] = 0
    output = model(input_ids, attention_mask)
    assert output.shape == (3, config.model.output_dim)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(3))
    counts = model.parameter_counts()
    assert counts["logical"] > counts["physical"]


def test_ternary_path_backpropagates() -> None:
    config, _ = load_experiment(Path("configs/smoke.yaml"))
    model = BrowserEncoder(config.model)
    assert set_quantization_strength(model, 1.0) == 5
    tokens = torch.randint(1, config.model.vocab_size, (2, 8))
    output = model(tokens, torch.ones_like(tokens))
    output.square().sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert len(ternary_health(model)) == 5
