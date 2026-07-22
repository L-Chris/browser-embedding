"""The single command-line entry point for training and export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from browser_embedding.config import find_project_root, load_experiment, resolve_path
from browser_embedding.export import export_checkpoint, inspect_header
from browser_embedding.preparation import (
    CacheRecipe,
    LegacyTernlightRecipe,
    TeacherRecipe,
    TokenizerRecipe,
    build_cache,
    encode_teacher,
    import_legacy_ternlight_cache,
    load_recipe,
    train_tokenizer,
    validate_cache,
)
from browser_embedding.preparation.contracts import resolve_recipe_path
from browser_embedding.runner import TrainingRunner

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
prepare_app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Build and audit immutable training artifacts.",
)
app.add_typer(prepare_app, name="prepare")


@prepare_app.command("cache")
def prepare_cache(
    recipe: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Build token memmaps and row metadata from canonical pair JSONL."""
    config, root = load_recipe(recipe, CacheRecipe)
    typer.echo(json.dumps(build_cache(config, root), ensure_ascii=False, indent=2))


@prepare_app.command("tokenizer")
def prepare_tokenizer(
    recipe: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Train the versioned tokenizer from a corpus that excludes evaluation data."""
    config, root = load_recipe(recipe, TokenizerRecipe)
    typer.echo(json.dumps(train_tokenizer(config, root), ensure_ascii=False, indent=2))


@prepare_app.command("teacher")
def prepare_teacher(
    recipe: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Generate resumable, normalized teacher targets for an existing cache."""
    config, root = load_recipe(recipe, TeacherRecipe)
    typer.echo(
        json.dumps(
            encode_teacher(config, root, requested_device=device),
            ensure_ascii=False,
            indent=2,
        )
    )


@prepare_app.command("import-ternlight")
def prepare_import_ternlight(
    recipe: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Import ternlight's bilingual token and teacher cache without recomputation."""
    config, root = load_recipe(recipe, LegacyTernlightRecipe)
    typer.echo(
        json.dumps(
            import_legacy_ternlight_cache(config, root),
            ensure_ascii=False,
            indent=2,
        )
    )


@prepare_app.command("validate")
def prepare_validate(
    cache: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    fast: Annotated[bool, typer.Option(help="Skip full-file SHA-256 verification.")] = False,
) -> None:
    """Audit cache shapes, hashes, row alignment and teacher normalization."""
    root = find_project_root(cache)
    cache_path = resolve_recipe_path(root, cache)
    typer.echo(
        json.dumps(validate_cache(cache_path, full_hash=not fast), ensure_ascii=False, indent=2)
    )


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
