from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from harness.localization.models import LCAGold, LCATask, normalize_lca_task
from harness.telemetry.tracing import get_langfuse_client


class LocalizationDatasetLoadError(RuntimeError):
    """Raised when a Langfuse localization dataset item cannot be validated."""


class LocalizationDataset(BaseModel):
    name: str
    version: str | None = None
    items: list[LCATask] = Field(default_factory=list)

    @property
    def entries(self) -> list[LCATask]:
        return self.items


def _defaults(
    dataset_name: str, dataset_config: str | None, dataset_split: str | None
) -> tuple[str, str, str]:
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


def _field(raw: object, name: str) -> object:
    if isinstance(raw, Mapping):
        return raw.get(name)
    return getattr(raw, name, None)


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise LocalizationDatasetLoadError(
        f"Langfuse dataset item {field_name} must be an object"
    )


def _expected_gold(expected_output: object) -> LCAGold:
    expected = _require_mapping(expected_output, "expected_output")
    try:
        return LCAGold.model_validate(expected)
    except Exception as exc:
        raise LocalizationDatasetLoadError(
            "Langfuse dataset item expected_output must contain valid changed_files gold"
        ) from exc


def task_from_langfuse_dataset_item(
    raw: object,
    *,
    dataset_name: str,
) -> LCATask:
    """
    Validate one Langfuse dataset item into the canonical localization task model.

    Langfuse owns task payloads in `input` and gold localization data in
    `expected_output`. Runtime code receives only the reconstructed `LCATask`.
    """

    input_payload = _require_mapping(_field(raw, "input"), "input")
    gold = _expected_gold(_field(raw, "expected_output"))
    task_payload = dict(input_payload)
    task_payload["gold"] = gold.model_dump()
    try:
        return LCATask.model_validate(task_payload)
    except Exception as exc:
        raise LocalizationDatasetLoadError(
            f"Langfuse dataset item input for {dataset_name!r} is not a valid LCATask payload"
        ) from exc


def langfuse_dataset_item_from_task(
    task: LCATask,
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the canonical Langfuse dataset item payload for one task."""

    return {
        "input": task.model_dump(mode="json", exclude={"gold"}),
        "expected_output": task.gold.model_dump(mode="json"),
        "metadata": dict(metadata) if metadata else {},
    }


def fetch_localization_dataset(
    name: str,
    version: str | None = None,
) -> list[LCATask]:
    """
    Fetch localization tasks from Langfuse dataset items.
    """
    client = get_langfuse_client()
    if not client:
        return []
    client_any: Any = client
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
        raw_items = (
            dataset.items()
            if callable(getattr(dataset, "items", None))
            else getattr(dataset, "items", [])
        )
    except Exception:
        raw_items = []

    return [
        task_from_langfuse_dataset_item(
            raw,
            dataset_name=name,
        )
        for raw in raw_items
    ]


def local_dataset(
    name: str,
    items: Sequence[Mapping[str, object] | LCATask],
    version: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
) -> LocalizationDataset:
    defaults = _defaults(name, dataset_config, dataset_split)
    normalized_items = [
        item if isinstance(item, LCATask) else normalize_lca_task(item, defaults=defaults)
        for item in items
    ]
    return LocalizationDataset(name=name, version=version, items=normalized_items)
