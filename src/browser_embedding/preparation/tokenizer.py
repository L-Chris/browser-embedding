"""Train the browser/runtime tokenizer from a versioned corpus."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from browser_embedding.preparation.contracts import (
    SPECIAL_TOKEN_IDS,
    TokenizerRecipe,
    resolve_recipe_path,
)
from browser_embedding.preparation.io import iter_jsonl, sha256_file, write_json


def _corpus_texts(path: Path, input_format: str) -> Iterator[str]:
    if input_format == "text":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if text := line.strip():
                    yield text
        return
    for record in iter_jsonl(path):
        if input_format == "jsonl_text":
            value = record.get("text")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path} jsonl_text rows require a non-empty text field")
            yield value.strip()
        elif input_format == "jsonl_pairs":
            for field in ("query", "document"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{path} jsonl_pairs rows require {field!r}")
                yield value.strip()
        else:  # guarded by TokenizerRecipe, retained for direct callers
            raise ValueError(f"unsupported tokenizer input format {input_format!r}")


def train_tokenizer(recipe: TokenizerRecipe, project_root: Path) -> dict[str, object]:
    """Train a 32k NFKC + ByteLevel Unigram tokenizer without eval leakage."""
    import tokenizers as tokenizers_package  # type: ignore[import-untyped]
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )

    input_path = resolve_recipe_path(project_root, recipe.input_path)
    output_path = resolve_recipe_path(project_root, recipe.output_path)
    manifest_path = resolve_recipe_path(project_root, recipe.manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=recipe.vocab_size,
        unk_token="[UNK]",
        special_tokens=list(SPECIAL_TOKEN_IDS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(_corpus_texts(input_path, recipe.input_format), trainer=trainer)
    actual_special = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKEN_IDS}
    if actual_special != SPECIAL_TOKEN_IDS:
        raise ValueError(f"trained tokenizer special IDs differ: {actual_special}")
    actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab != recipe.vocab_size:
        raise ValueError(f"trained vocabulary {actual_vocab} != requested {recipe.vocab_size}")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", SPECIAL_TOKEN_IDS["[CLS]"]), ("[SEP]", 3)],
    )
    tokenizer.save(str(output_path), pretty=False)
    bytes_written = output_path.stat().st_size
    if bytes_written > recipe.asset_budget_bytes:
        raise ValueError(
            f"tokenizer asset {bytes_written:,} exceeds budget {recipe.asset_budget_bytes:,}"
        )

    lengths: list[int] = []
    unknown_tokens = 0
    audited_texts = 0
    for text in _corpus_texts(input_path, recipe.input_format):
        encoding = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(encoding.ids))
        unknown_tokens += encoding.ids.count(SPECIAL_TOKEN_IDS["[UNK]"])
        audited_texts += 1
        if audited_texts == 10_000:
            break
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": output_path.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "candidate",
        "evaluation_data_in_training_corpus": False,
        "tokenizers_version": tokenizers_package.__version__,
        "model": "Unigram",
        "normalizer": "NFKC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=false)",
        "vocab_size": actual_vocab,
        "special_token_ids": SPECIAL_TOKEN_IDS,
        "bytes": bytes_written,
        "sha256": sha256_file(output_path),
        "corpus": {
            "name": recipe.corpus.name,
            "revision": recipe.corpus.revision,
            "license": recipe.corpus.license,
            "path": str(recipe.input_path),
            "sha256": sha256_file(input_path),
            "format": recipe.input_format,
        },
        "audit": {
            "texts": audited_texts,
            "unknown_tokens": unknown_tokens,
            "length_mean": sum(lengths) / max(len(lengths), 1),
            "length_max": max(lengths, default=0),
        },
    }
    write_json(manifest_path, manifest)
    return manifest
