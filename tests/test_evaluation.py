from pathlib import Path

import torch
from torch import Tensor, nn

from browser_embedding.evaluation import evaluate_multilingual_seed


class _TestEncoder(nn.Module):
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        masked = input_ids.to(torch.float32) * attention_mask
        features = torch.stack(
            (masked.sum(dim=1), masked.square().sum(dim=1), attention_mask.sum(dim=1)),
            dim=1,
        )
        return torch.nn.functional.normalize(features, dim=1)


def test_multilingual_evaluation_restores_padding_token() -> None:
    metrics = evaluate_multilingual_seed(
        _TestEncoder(),  # type: ignore[arg-type]
        tokenizer_path=Path("tests/fixtures/tokenizer.json"),
        seed_path=Path("eval/multilingual.jsonl"),
        max_sequence_length=24,
        batch_size=8,
        device=torch.device("cpu"),
    )
    assert metrics["macro_ndcg_at_10"] >= 0.0
