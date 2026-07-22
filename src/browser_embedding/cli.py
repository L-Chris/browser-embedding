"""The single command-line entry point for training and export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from browser_embedding.config import load_experiment, resolve_path
from browser_embedding.export import export_checkpoint, inspect_header
from browser_embedding.runner import TrainingRunner

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def inspect(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Validate an experiment and print its browser budget before training."""
    experiment, _ = load_experiment(config)
    estimate = experiment.deployment.estimate(experiment.model)
    typer.echo(
        json.dumps(
            {
                "model": experiment.model.model_dump(mode="json"),
                "deployment": experiment.deployment.model_dump(mode="json"),
                "estimate": estimate.as_dict(),
            },
            indent=2,
        )
    )


@app.command()
def train(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    resume: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    max_batches: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Run data → train → validate → checkpoint → optional BEM2 export."""
    experiment, root = load_experiment(config)
    resume_path = resolve_path(root, resume) if resume is not None else None
    best = TrainingRunner(experiment, root, resume=resume_path, max_batches=max_batches).run()
    typer.echo(f"best_checkpoint={best}")


@app.command(name="export")
def export_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Export a versioned training checkpoint to BEM2."""
    manifest = export_checkpoint(checkpoint.resolve(), output.resolve())
    typer.echo(json.dumps(manifest, ensure_ascii=False, indent=2))


@app.command(name="inspect-model")
def inspect_model(
    model: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Validate the header, size and SHA-256 of a BEM2 artifact."""
    typer.echo(json.dumps(inspect_header(model), ensure_ascii=False, indent=2))
