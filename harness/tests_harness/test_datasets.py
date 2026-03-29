from __future__ import annotations

from harness.observability import datasets


def test_normalize_dataset_item_generates_id_when_missing():
    item = {"repo": "r", "symbol": "s"}
    normalized = datasets.normalize_dataset_item(item)
    assert normalized["repo"] == "r"
    assert normalized["symbol"] == "s"
    assert isinstance(normalized.get("id"), str)
