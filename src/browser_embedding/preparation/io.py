"""Deterministic hashing and atomic metadata writes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))
