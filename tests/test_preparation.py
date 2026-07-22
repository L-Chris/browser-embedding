from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import browser_embedding.preparation.teacher as teacher_module
from browser_embedding.config import load_experiment
from browser_embedding.data import MemmapPairDataset
from browser_embedding.preparation.cache import build_cache, validate_cache
from browser_embedding.preparation.contracts import (
    CacheRecipe,
    LegacyTernlightRecipe,
    TeacherRecipe,
)
from browser_embedding.preparation.io import sha256_file
from browser_embedding.preparation.legacy_ternlight import import_legacy_ternlight_cache
from browser_embedding.preparation.teacher import encode_teacher

tokenizers = pytest.importorskip("tokenizers")


def _write_pairs(path: Path) -> None:
    rows = [
        ("a", "train", "en_en", "reset password", "reset your account password"),
        ("b", "train", "zh_zh", "如何退款", "可以在订单页面申请退款"),
        ("c", "train", "zh_en", "怎么修改 address", "Update the address in settings"),
        ("d", "train", "en_zh", "change my plan", "在账户页面修改套餐"),
        ("e", "validation", "mixed_en", "API 请求失败", "Retry the failed API request"),
        ("f", "test", "mixed_zh", "download 在哪里", "下载入口位于文件页面"),
        ("duplicate", "train", "en_en", "reset password", "reset your account password"),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for identifier, split, variant, query, document in rows:
            handle.write(
                json.dumps(
                    {
                        "id": identifier,
                        "group_id": identifier,
                        "query": query,
                        "document": document,
                        "variant": variant,
                        "source": "fixture",
                        "source_revision": "fixture-v1",
                        "source_license": "CC0-1.0",
                        "split": split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_build_validate_and_load_memmap_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    cache_dir = tmp_path / "cache"
    _write_pairs(pairs_path)
    recipe = CacheRecipe(
        schema_version=1,
        input_path=pairs_path,
        output_dir=cache_dir,
        tokenizer_path=Path("assets/tokenizer.json"),
        max_sequence_length=128,
        vocab_size=32_000,
        batch_size=3,
    )
    manifest = build_cache(recipe, Path.cwd())
    assert manifest["samples"] == 6
    assert manifest["duplicates_removed"] == 1
    assert manifest["tokenizer"]["missing_role_tokens"] == {"query": 0, "document": 0}
    assert validate_cache(cache_dir)["status"] == "ok"

    config, _ = load_experiment(Path("configs/multilingual.yaml"))

    class FakeTeacher:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_sentence_embedding_dimension(self) -> int:
            return config.model.output_dim

        def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
            output = np.zeros((len(texts), config.model.output_dim), dtype=np.float32)
            for row, text in enumerate(texts):
                column = int(sha256(text.encode()).hexdigest()[:8], 16) % output.shape[1]
                output[row, column] = 1
            return output

    original_import = teacher_module.importlib.import_module

    def fake_import(name: str) -> object:
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=FakeTeacher, __version__="test")
        return original_import(name)

    monkeypatch.setattr(teacher_module.importlib, "import_module", fake_import)
    teacher_recipe = TeacherRecipe(
        schema_version=1,
        cache_dir=cache_dir,
        key="fixture",
        model_id="fixture/model",
        revision="0" * 40,
        license="CC0-1.0",
        output_dimension=config.model.output_dim,
        chunk_size=2,
        batch_size=2,
    )
    teacher_manifest = encode_teacher(teacher_recipe, Path.cwd(), requested_device="cpu")
    assert teacher_manifest["shape"] == [6, 2, config.model.output_dim]

    report = validate_cache(cache_dir)
    assert report["teachers"]["fixture"]["norm_max_error"] < 0.01
    dataset = MemmapPairDataset(cache_dir, "fixture", "validation", config.model)
    assert len(dataset) == 1
    assert dataset[0]["variant"] == "mixed_en"


def test_candidate_tokenizer_matches_manifest() -> None:
    from tokenizers import Tokenizer

    path = Path("assets/tokenizer.json")
    manifest = json.loads(Path("assets/tokenizer.manifest.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(path))
    assert tokenizer.get_vocab_size(with_added_tokens=True) == 32_000
    assert manifest["file"]["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert manifest["quality_claims_allowed"] is False
    assert {
        token: tokenizer.token_to_id(token)
        for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[QRY]", "[DOC]")
    } == {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[QRY]": 4, "[DOC]": 5}


def test_import_legacy_ternlight_cache(tmp_path: Path) -> None:
    from tokenizers import Tokenizer

    source = tmp_path / "legacy"
    shared = source / "shared"
    teacher_dir = source / "minilm"
    shared.mkdir(parents=True)
    teacher_dir.mkdir()
    rows = [
        ("a", "train", "en_en", "reset password", "reset your account password"),
        ("b", "validation", "zh_zh", "如何退款", "可以在订单页面申请退款"),
        ("c", "test", "mixed_en", "API 请求失败", "Retry the failed API request"),
    ]
    with (shared / "selection.jsonl").open("w", encoding="utf-8") as handle:
        for identifier, split, variant, query, document in rows:
            handle.write(
                json.dumps(
                    {
                        "id": identifier,
                        "group_id": identifier,
                        "query": query,
                        "positive": document,
                        "variant": variant,
                        "source": "fixture",
                        "split": split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    tokenizer_path = Path("assets/tokenizer.json").resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128, pad_id=0, pad_token="[PAD]")
    values = [f"[QRY] {row[3]}" for row in rows] + [f"[DOC] {row[4]}" for row in rows]
    encoded = tokenizer.encode_batch(values, add_special_tokens=True)
    input_ids = np.zeros((len(rows), 2, 128), dtype=np.uint32)
    attention_mask = np.zeros_like(input_ids, dtype=np.uint8)
    for index, encoding in enumerate(encoded):
        role = 0 if index < len(rows) else 1
        row = index if role == 0 else index - len(rows)
        input_ids[row, role] = encoding.ids
        attention_mask[row, role] = encoding.attention_mask
    np.save(shared / "input_ids.npy", input_ids)
    np.save(shared / "attention_mask.npy", attention_mask)
    np.save(shared / "split_codes.npy", np.asarray([0, 1, 2], dtype=np.uint8))
    (shared / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": len(rows),
                "max_length": 128,
                "vocab_size": 32_000,
                "selection_sha256": "legacy-selection",
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    corpus_manifest = tmp_path / "corpus.manifest.json"
    corpus_manifest.write_text(
        json.dumps({"source_revisions": {"fixture": "fixture-revision"}}),
        encoding="utf-8",
    )
    teacher = np.zeros((len(rows), 2, 384), dtype=np.float16)
    teacher[..., 0] = 1
    np.save(teacher_dir / "embeddings.npy", teacher)
    (teacher_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "teacher_key": "minilm",
                "model_id": "fixture/teacher",
                "model_revision": "0" * 40,
                "query_prefix": "",
                "document_prefix": "",
                "selection_sha256": "legacy-selection",
                "shape": list(teacher.shape),
                "dtype": "float16",
                "sha256": sha256_file(teacher_dir / "embeddings.npy"),
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "imported"
    recipe = LegacyTernlightRecipe(
        schema_version=1,
        source_cache_dir=source,
        source_corpus_manifest=corpus_manifest,
        output_dir=output,
        tokenizer_path=tokenizer_path,
        source_teacher_key="minilm",
        teacher_key="fixture-teacher",
        teacher_license="Apache-2.0",
        source_licenses={"fixture": "CC0-1.0"},
        link_mode="copy",
        verify_tokenizer=True,
    )
    manifest = import_legacy_ternlight_cache(recipe, Path.cwd())
    assert manifest["samples"] == len(rows)
    assert manifest["migration"]["tokenizer_cache_fully_verified"] is True
    assert manifest["migration"]["license_audit_required"] is False
    assert validate_cache(output)["teachers"]["fixture-teacher"]["norm_max_error"] == 0


def test_teacher_recipe_requires_immutable_revision() -> None:
    with pytest.raises(ValueError, match="revision"):
        TeacherRecipe(
            schema_version=1,
            cache_dir=Path("cache"),
            key="teacher",
            model_id="organization/model",
            revision="main",
            license="Apache-2.0",
        )
