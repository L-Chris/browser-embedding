"""Versioned training checkpoint I/O."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

CHECKPOINT_KIND = "browser-embedding-training-checkpoint"
CHECKPOINT_VERSION = 1


def capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "dataloader": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["dataloader"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"kind": CHECKPOINT_KIND, "schema_version": CHECKPOINT_VERSION, **payload}, temporary
    )
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"{path} is not a browser-embedding checkpoint")
    if checkpoint.get("schema_version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint schema {checkpoint.get('schema_version')!r}")
    return cast(dict[str, Any], checkpoint)
