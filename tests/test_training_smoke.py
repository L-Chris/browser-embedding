from pathlib import Path

from browser_embedding.checkpoint import load_checkpoint
from browser_embedding.config import RunConfig, load_experiment
from browser_embedding.export import inspect_header
from browser_embedding.runner import TrainingRunner


def test_complete_training_pipeline(tmp_path: Path) -> None:
    config, root = load_experiment(Path("configs/smoke.yaml"))
    config = config.model_copy(
        update={
            "run": RunConfig(name="test-smoke", output_dir=tmp_path / "run", export_on_finish=True)
        }
    )
    best_path = TrainingRunner(config, root, max_batches=1).run()
    checkpoint = load_checkpoint(best_path)
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 1
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    header = inspect_header(tmp_path / "run" / "model.bem")
    assert header["output_dim"] == config.model.output_dim
