from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from harness.localization.models import LCAGold, LCATask, normalize_lca_task
from harness.log import bind_logger, get_logger
from harness.telemetry.tracing import get_langfuse_client

logger = get_logger(__name__)


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


def _item_identity(raw: object, task_payload: Mapping[str, object] | None = None) -> str:
    item_id = _field(raw, "id")
    if isinstance(item_id, str) and item_id:
        return item_id
    metadata = _field(raw, "metadata")
    if isinstance(metadata, Mapping):
        for key in ("canonical_task_identity", "task_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    if task_payload:
        identity = task_payload.get("identity")
        if isinstance(identity, Mapping):
            repo_owner = identity.get("repo_owner")
            repo_name = identity.get("repo_name")
            base_sha = identity.get("base_sha")
            if repo_owner and repo_name and base_sha:
                return f"{repo_owner}/{repo_name}@{base_sha}"
    return "<unknown>"


def _repo_from_payload_or_identity(task_payload: Mapping[str, object], task: LCATask) -> str | None:
    repo = task_payload.get("repo")
    if isinstance(repo, str) and repo.strip():
        return repo.strip()
    metadata = task_payload.get("metadata")
    if isinstance(metadata, Mapping):
        metadata_repo = metadata.get("repo")
        if isinstance(metadata_repo, str) and metadata_repo.strip():
            return metadata_repo.strip()
    if task.identity.repo_owner and task.identity.repo_name:
        return f"{task.identity.repo_owner}/{task.identity.repo_name}"
    return None


def _validate_runtime_task(task: LCATask, *, dataset_name: str, item_identity: str) -> LCATask:
    missing: list[str] = []
    if not task.repo:
        missing.append("repo")
    if not task.identity.repo_owner:
        missing.append("identity.repo_owner")
    if not task.identity.repo_name:
        missing.append("identity.repo_name")
    if not task.identity.base_sha:
        missing.append("identity.base_sha")
    if not task.context.issue_title and not task.context.issue_body:
        missing.append("context.issue_title or context.issue_body")
    if not task.gold.changed_files:
        missing.append("expected_output.changed_files")
    if missing:
        fields = ", ".join(f"`{field}`" for field in missing)
        raise LocalizationDatasetLoadError(
            f"selected Langfuse dataset item {item_identity} from {dataset_name!r} is missing required field(s) {fields}; "
            "dataset sync schema is incomplete for runtime execution"
        )
    return task


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
    item_identity = _item_identity(raw, task_payload)
    try:
        task = LCATask.model_validate(task_payload)
    except Exception as exc:
        raise LocalizationDatasetLoadError(
            f"Langfuse dataset item input for {dataset_name!r} item={item_identity} is not a valid LCATask payload"
        ) from exc
    repo = _repo_from_payload_or_identity(task_payload, task)
    if repo and task.repo != repo:
        task = task.with_repo(repo)
    return _validate_runtime_task(task, dataset_name=dataset_name, item_identity=item_identity)


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

    tasks: list[LCATask] = []
    rejected = 0
    for raw in raw_items:
        try:
            tasks.append(task_from_langfuse_dataset_item(raw, dataset_name=name))
        except LocalizationDatasetLoadError:
            rejected += 1
            raise
    bind_logger(logger, dataset=name, version=version).info(
        "langfuse_dataset_converted",
        task_count=len(tasks),
        rejected=rejected,
    )
    return tasks


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
