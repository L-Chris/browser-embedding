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
from browser_embedding.preparation.contracts import CacheRecipe, TeacherRecipe
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
