from pathlib import Path

import pytest

from browser_embedding.config import load_experiment
from browser_embedding.export import export_model, inspect_header
from browser_embedding.model import BrowserEncoder


def test_export_matches_budget_and_round_trips_header(tmp_path: Path) -> None:
    config, _ = load_experiment(Path("configs/smoke.yaml"))
    model = BrowserEncoder(config.model)
    output = tmp_path / "model.bem"
    manifest = export_model(model, config, output)
    header = inspect_header(output)
    assert manifest["bytes"] == config.deployment.estimate(config.model).packed_model_bytes
    assert header["bytes"] == manifest["bytes"]
    assert header["matryoshka_dims"] == list(config.model.matryoshka_dims)
    assert len(manifest["sections"]) == 12


def test_corrupt_export_is_rejected(tmp_path: Path) -> None:
    config, _ = load_experiment(Path("configs/smoke.yaml"))
    output = tmp_path / "model.bem"
    export_model(BrowserEncoder(config.model), config, output)
    binary = bytearray(output.read_bytes())
    binary[80] ^= 1
    output.write_bytes(binary)
    with pytest.raises(ValueError, match="checksum"):
        inspect_header(output)
