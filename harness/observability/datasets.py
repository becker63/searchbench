from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, Field

from harness.localization.models import LCATask, normalize_lca_task

from .langfuse import get_langfuse_client


class LocalizationDataset(BaseModel):
    name: str
    version: str | None = None
    items: list[LCATask] = Field(default_factory=list)

    @property
    def entries(self) -> list[LCATask]:
        return self.items


def _defaults(dataset_name: str, dataset_config: str | None, dataset_split: str | None) -> tuple[str, str, str]:
    return (
        dataset_name,
        dataset_config or "unspecified-config",
        dataset_split or "unspecified-split",
    )


def normalize_localization_task(
    item: Mapping[str, object] | LCATask,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
) -> LCATask:
    """
    Normalize an arbitrary mapping into an LCATask using dataset defaults where provided.
    """
    if isinstance(item, LCATask):
        return item
    defaults = None
    if dataset_name:
        defaults = _defaults(dataset_name, dataset_config, dataset_split)
    return normalize_lca_task(item, defaults=defaults)


def fetch_localization_dataset(
    name: str,
    version: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
) -> list[LCATask]:
    """
    Fetch localization tasks from Langfuse dataset items.
    Expects each item to include LCA identity fields and changed_files; legacy symbol fields are not used.
    """
    client = get_langfuse_client()
    if not client:
        return []
    client_any = cast(Any, client)
    dataset: Any = None
    try:
        dataset = (
            client_any.get_dataset(name=name, version=version)
            if hasattr(client_any, "get_dataset")
            else client_any.dataset(name=name, version=version)
        )
    except Exception:
        dataset = None
    if dataset is None:
        return []
    try:
        raw_items = dataset.items() if callable(getattr(dataset, "items", None)) else getattr(dataset, "items", [])
    except Exception:
        raw_items = []
    tasks: list[LCATask] = []
    defaults = _defaults(name, dataset_config, dataset_split)
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        # Langfuse typically stores payload under "input"
        input_section = raw.get("input") if isinstance(raw.get("input"), Mapping) else raw
        try:
            tasks.append(normalize_lca_task(cast(Mapping[str, object], input_section), defaults=defaults))
        except Exception:
            continue
    return tasks


def local_dataset(
    name: str,
    items: Sequence[Mapping[str, object] | LCATask],
    version: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
) -> LocalizationDataset:
    defaults = _defaults(name, dataset_config, dataset_split)
    normalized_items = [normalize_lca_task(item, defaults=defaults) for item in items]  # type: ignore[arg-type]
    return LocalizationDataset(name=name, version=version, items=normalized_items)
