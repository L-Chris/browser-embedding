"""Compare the BEM2 Rust graph against the materialized PyTorch graph."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from browser_embedding.config import load_experiment
from browser_embedding.export import export_model, materialize_deployment_model
from browser_embedding.model import BrowserEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--cargo", default="cargo")
    arguments = parser.parse_args()

    config, root = load_experiment(arguments.config)
    torch.manual_seed(config.seed)
    model = BrowserEncoder(config.model).eval()
    reference = materialize_deployment_model(model, config)
    sequence = min(12, config.model.max_sequence_length)
    input_ids = torch.randint(6, config.model.vocab_size, (1, sequence))
    input_ids[0, 0] = 4
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -2:] = 0
    dimension = config.model.matryoshka_dims[-1]
    with torch.inference_mode():
        expected = reference.truncate(reference(input_ids, attention_mask), dimension)[0].numpy()

    with tempfile.TemporaryDirectory(prefix="browser-embedding-parity-") as directory:
        model_path = Path(directory) / "model.bem"
        export_model(model, config, model_path)
        result = subprocess.run(
            [
                arguments.cargo,
                "run",
                "--quiet",
                "-p",
                "browser-embedding-core",
                "--example",
                "embed",
                "--",
                str(model_path),
                ",".join(str(value) for value in input_ids[0].tolist()),
                ",".join(str(value) for value in attention_mask[0].tolist()),
                str(dimension),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    actual = np.fromstring(result.stdout.strip(), sep=",", dtype=np.float32)
    difference = np.abs(expected - actual)
    print(
        f"dimension={dimension} max_abs_diff={difference.max():.3e} "
        f"mean_abs_diff={difference.mean():.3e}"
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    main()
