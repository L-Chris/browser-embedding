import json
from pathlib import Path


def test_multilingual_seed_covers_all_required_fields() -> None:
    path = Path("eval/multilingual.jsonl")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    required = {"id", "domain", "query_en", "query_zh", "query_mixed", "doc_en", "doc_zh"}
    assert len(records) >= 12
    assert all(required <= record.keys() for record in records)
    assert len({record["id"] for record in records}) == len(records)
    assert all(
        record["query_mixed"] not in {record["query_en"], record["query_zh"]} for record in records
    )
