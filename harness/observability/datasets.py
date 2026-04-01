from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, Field, model_validator

from .langfuse import get_langfuse_client


def _stable_id(repo: str, symbol: str) -> str:
    return hashlib.sha256(f"{repo}::{symbol}".encode()).hexdigest()


class DatasetItem(BaseModel):
    repo: str
    symbol: str
    metadata: dict[str, object] = Field(default_factory=dict)
    id: str | None = None

    @model_validator(mode="after")
    def _ensure_id(self) -> "DatasetItem":
        if not self.id:
            self.id = _stable_id(self.repo, self.symbol)
        return self


class Dataset(BaseModel):
    name: str
    items: list[DatasetItem] = Field(default_factory=list)

    @property
    def entries(self) -> list[DatasetItem]:
        return self.__dict__.get("items", [])


def map_task_to_dataset_item(task: Mapping[str, object] | DatasetItem) -> DatasetItem:
    if isinstance(task, DatasetItem):
        return task
    repo = task.get("repo")
    symbol = task.get("symbol")
    if not isinstance(repo, str) or not isinstance(symbol, str):
        raise ValueError("Task must include 'repo' and 'symbol'")
    metadata = {k: v for k, v in task.items() if k not in {"repo", "symbol"}}
    return DatasetItem(repo=repo, symbol=symbol, metadata=metadata)


def normalize_dataset_item(item: Mapping[str, object] | DatasetItem) -> DatasetItem:
    if isinstance(item, DatasetItem):
        return item
    repo_obj = item.get("repo")
    symbol_obj = item.get("symbol")
    if not isinstance(repo_obj, str) or not isinstance(symbol_obj, str):
        raise ValueError("Dataset item missing repo/symbol")
    metadata_obj = item.get("metadata")
    metadata: dict[str, object] = {}
    if isinstance(metadata_obj, Mapping):
        metadata = {str(k): v for k, v in metadata_obj.items()}
    normalized = DatasetItem(repo=repo_obj, symbol=symbol_obj, metadata=metadata)
    item_id = item.get("id")
    if isinstance(item_id, str):
        normalized.id = item_id
    elif item_id is not None:
        normalized.id = str(item_id)
    return normalized


def fetch_dataset_items(name: str, version: str | None = None) -> list[DatasetItem]:
    """
    Fetch dataset items from Langfuse. Falls back to empty list if unavailable.
    """
    client = get_langfuse_client()
    if not client:
        return []
    client_any = cast(Any, client)
    dataset: Any = None
    try:
        if hasattr(client_any, "get_dataset"):
            dataset = client_any.get_dataset(name=name, version=version)  # type: ignore[arg-type]
        elif hasattr(client_any, "dataset"):
            dataset = client_any.dataset(name=name, version=version)  # type: ignore[arg-type]
    except Exception:
        dataset = None
    if dataset is None:
        return []
    try:
        raw_items = dataset.items() if callable(getattr(dataset, "items", None)) else getattr(dataset, "items", [])
    except Exception:
        raw_items = []
    items: list[DatasetItem] = []
    for item in raw_items:
        if isinstance(item, Mapping):
            task_meta = cast(Mapping[str, object], item)
            input_section = task_meta.get("input")
            if not isinstance(input_section, Mapping):
                input_section = {}
            repo = task_meta.get("repo") or input_section.get("repo")
            symbol = task_meta.get("symbol") or input_section.get("symbol")
            if not isinstance(repo, str) or not isinstance(symbol, str):
                continue
            meta = task_meta.get("metadata")
            meta_map: Mapping[str, object] = input_section if isinstance(input_section, Mapping) else {}
            if isinstance(meta, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
                meta_map = meta
            try:
                mapped = map_task_to_dataset_item({"repo": repo, "symbol": symbol, **{str(k): v for k, v in meta_map.items()}})
                item_id = task_meta.get("id")
                if isinstance(item_id, str):
                    mapped.id = item_id
                elif item_id is not None:
                    mapped.id = str(item_id)
                items.append(mapped)
            except Exception:
                continue
    return items


def local_dataset(name: str, items: Sequence[Mapping[str, object] | DatasetItem]) -> Dataset:
    normalized_items = [normalize_dataset_item(item) for item in items]
    return Dataset(name=name, items=normalized_items)
