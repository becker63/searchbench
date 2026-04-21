from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness.log import bind_logger, get_logger
from harness.localization.models import LCATask, normalize_lca_task
from harness.telemetry.hosted.datasets import langfuse_dataset_item_from_task
from harness.telemetry.tracing import get_langfuse_client

logger = get_logger(__name__)
SYNC_PROGRESS_INTERVAL = 250


class HFDatasetLoadError(RuntimeError):
    """
    Error raised when HF dataset loading fails.
    `category` is used to surface failure class in CLI/telemetry.
    """

    def __init__(self, message: str, category: str = "unknown"):
        super().__init__(message)
        self.category = category


class HFDatasetSyncResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[dict[str, object]] = Field(default_factory=list)
    source_items: int
    existing_items: int
    created_items: int
    skipped_items: int
    failed_items: int = 0


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def compute_lca_task_identity(task: LCATask) -> str:
    """
    Deterministic Langfuse dataset item id for one logical LCA example.

    Identity is defined by the upstream example itself: repo owner, repo name,
    base SHA, and issue/pull identity when available. If no issue or pull URL is
    present, issue title/body are used as the fallback discriminator. Dataset
    config, split, revision, and target Langfuse dataset are provenance only and
    do not participate in this id.
    """

    payload = {
        "repo_owner": task.identity.repo_owner,
        "repo_name": task.identity.repo_name,
        "base_sha": task.identity.base_sha,
        "issue_url": task.identity.issue_url,
        "pull_url": task.identity.pull_url,
        "issue_title": task.context.issue_title if not (task.identity.issue_url or task.identity.pull_url) else None,
        "issue_body": task.context.issue_body if not (task.identity.issue_url or task.identity.pull_url) else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"lca-{hashlib.sha256(encoded).hexdigest()[:32]}"


def _require_dependency() -> Any:
    """
    Import the HF datasets library lazily to avoid mandatory networked deps in hermetic runs.
    """
    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised via failure classification
        raise HFDatasetLoadError(
            "Hugging Face datasets dependency is missing; install `datasets` to enable HF loading",
            category="dependency-missing",
        ) from exc
    return datasets


def _validate_rows(rows: Iterable[Mapping[str, object]]) -> Sequence[Mapping[str, object]]:
    """
    Validate that required LCA fields exist before normalization.
    """
    validated: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise HFDatasetLoadError("HF row is not a mapping", category="schema")
        required = [
            "repo_owner",
            "repo_name",
            "base_sha",
            "issue_title",
            "issue_body",
            "changed_files",
        ]
        missing = [field for field in required if field not in row or row[field] in (None, "")]
        if missing:
            raise HFDatasetLoadError(f"HF row missing required fields: {missing}", category="schema")
        validated.append(row)
    return validated


def fetch_hf_localization_dataset(
    name: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> list[LCATask]:
    """
    Fetch JetBrains-Research/lca-bug-localization tasks from Hugging Face, normalize to LCATask.
    Networked access is opt-in; if the dependency is missing or auth fails, a categorized HFDatasetLoadError is raised.
    """
    started_at = time.perf_counter()
    load_logger = bind_logger(
        logger,
        dataset=name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        revision=revision,
    )
    load_logger.info(
        "hf_dataset_load_started",
    )
    datasets = _require_dependency()
    try:
        hf_dataset = datasets.load_dataset(
            name,
            name if dataset_config is None else dataset_config,
            split=dataset_split,
            revision=revision,
            use_auth_token=token,
        )
    except datasets.utils.AuthenticationError as exc:  # type: ignore[attr-defined]
        raise HFDatasetLoadError("HF auth failed; set token via env/config", category="auth") from exc
    except FileNotFoundError as exc:
        raise HFDatasetLoadError("HF dataset/config/split not found", category="missing-dataset") from exc
    except Exception as exc:  # pragma: no cover - network issues
        raise HFDatasetLoadError(f"HF dataset load failed: {exc}", category="load") from exc

    try:
        rows = _validate_rows(hf_dataset)  # type: ignore[arg-type]
    except HFDatasetLoadError:
        raise
    except Exception as exc:
        raise HFDatasetLoadError(f"HF dataset validation failed: {exc}", category="schema") from exc

    defaults = (
        name,
        dataset_config or "unspecified-config",
        dataset_split or "unspecified-split",
    )
    tasks: list[LCATask] = []
    for row in rows:
        try:
            tasks.append(normalize_lca_task(row, defaults=defaults))
        except Exception as exc:
            raise HFDatasetLoadError(f"HF row normalization failed: {exc}", category="schema") from exc
    load_logger.info(
        "hf_dataset_load_completed",
        task_count=len(tasks),
        elapsed=_elapsed(started_at),
    )
    return tasks


def hf_source_metadata(
    *,
    dataset_name: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    revision: str | None = None,
) -> dict[str, object]:
    """Return provenance metadata for HF-to-Langfuse sync."""

    return {
        "source": "huggingface",
        "source_dataset": dataset_name,
        "source_config": dataset_config,
        "source_split": dataset_split,
        "source_revision": revision,
    }


def build_langfuse_items_from_hf_tasks(
    tasks: Sequence[LCATask],
    *,
    dataset_name: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    revision: str | None = None,
) -> list[dict[str, object]]:
    """Normalize HF-loaded tasks into canonical Langfuse dataset item payloads."""

    metadata = hf_source_metadata(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        revision=revision,
    )
    items: list[dict[str, object]] = []
    for task in tasks:
        repo = task.repo or f"{task.identity.repo_owner}/{task.identity.repo_name}"
        sync_task = task.with_repo(repo)
        items.append(
            langfuse_dataset_item_from_task(
                sync_task,
                metadata={
                    **metadata,
                    "canonical_task_identity": compute_lca_task_identity(task),
                    "task_id": sync_task.task_id,
                    "repo": repo,
                    "repo_owner": sync_task.identity.repo_owner,
                    "repo_name": sync_task.identity.repo_name,
                    "base_sha": sync_task.identity.base_sha,
                },
            )
            | {"id": compute_lca_task_identity(task)}
        )
    return items


def _get_field(obj: object, field: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _require_client_method(client: object, method_name: str):
    method = getattr(client, method_name, None)
    if not callable(method):
        raise HFDatasetLoadError(
            f"Langfuse client does not expose documented `{method_name}` needed for HF sync",
            category="langfuse-client",
        )
    return method


def _get_dataset(client: object, dataset_name: str) -> object:
    get_dataset = _require_client_method(client, "get_dataset")
    try:
        return get_dataset(name=dataset_name, fetch_items_page_size=100)
    except TypeError:
        return get_dataset(dataset_name)


def _ensure_langfuse_dataset(client: object, dataset_name: str) -> object:
    started_at = time.perf_counter()
    dataset_logger = bind_logger(logger, dataset=dataset_name)
    dataset_logger.info("langfuse_dataset_resolve_started")
    try:
        dataset = _get_dataset(client, dataset_name)
        dataset_logger.info(
            "langfuse_dataset_resolve_completed",
            elapsed=_elapsed(started_at),
        )
        return dataset
    except Exception as get_exc:  # noqa: BLE001
        dataset_logger.info("langfuse_dataset_create_attempt_started")
        create_dataset = _require_client_method(client, "create_dataset")
        try:
            create_dataset(
                name=dataset_name,
                metadata={"source": "harness.sync_hf"},
            )
        except Exception as create_exc:  # noqa: BLE001
            raise HFDatasetLoadError(
                f"Unable to fetch or create Langfuse dataset '{dataset_name}': {create_exc}",
                category="langfuse-client",
            ) from get_exc
        dataset = _get_dataset(client, dataset_name)
        dataset_logger.info(
            "langfuse_dataset_create_completed",
            elapsed=_elapsed(started_at),
        )
        return dataset


def _stable_identity_from_item(item: object) -> str | None:
    metadata = _get_field(item, "metadata")
    if isinstance(metadata, Mapping):
        canonical_identity = metadata.get("canonical_task_identity")
        if isinstance(canonical_identity, str) and canonical_identity:
            return canonical_identity
    item_id = _get_field(item, "id")
    if isinstance(item_id, str) and item_id:
        return item_id
    return None


def _fetch_existing_langfuse_items(
    client: object,
    *,
    dataset_name: str,
) -> list[object]:
    started_at = time.perf_counter()
    dataset_logger = bind_logger(logger, dataset=dataset_name)
    dataset_logger.info("langfuse_dataset_items_fetch_started")
    dataset = _ensure_langfuse_dataset(client, dataset_name)
    items = _get_field(dataset, "items")
    if items is None:
        raise HFDatasetLoadError(
            "Langfuse dataset fetch did not expose `items`; cannot compare existing dataset item identities",
            category="langfuse-client",
        )
    if not isinstance(items, Iterable):
        raise HFDatasetLoadError(
            "Langfuse dataset `items` is not iterable; cannot compare existing dataset item identities",
            category="langfuse-client",
        )
    existing_items = list(items)
    dataset_logger.info(
        "langfuse_dataset_items_fetch_completed",
        item_count=len(existing_items),
        elapsed=_elapsed(started_at),
    )
    return existing_items


def _index_existing_item_identities(items: Sequence[object]) -> set[str]:
    started_at = time.perf_counter()
    total = len(items)
    logger.info(
        "langfuse_item_identity_index_started",
        item_count=total,
    )
    identities: set[str] = set()
    for index, item in enumerate(items, start=1):
        identity = _stable_identity_from_item(item)
        if identity is not None:
            identities.add(identity)
        if index % SYNC_PROGRESS_INTERVAL == 0 or index == total:
            logger.info(
                "langfuse_item_identity_index_progress",
                processed=index,
                total=total,
                unique_identities=len(identities),
            )
    logger.info(
        "langfuse_item_identity_index_completed",
        unique_identities=len(identities),
        source_items=total,
        elapsed=_elapsed(started_at),
    )
    return identities


def _create_langfuse_dataset_item(
    client: object,
    *,
    dataset_name: str,
    item: Mapping[str, object],
) -> None:
    create_item = _require_client_method(client, "create_dataset_item")
    create_item(
        dataset_name=dataset_name,
        id=str(item["id"]),
        input=item["input"],
        expected_output=item["expected_output"],
        metadata=item["metadata"],
    )


def sync_hf_localization_dataset_to_langfuse(
    *,
    hf_dataset: str,
    langfuse_dataset: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> HFDatasetSyncResult:
    """
    Load HF localization rows, normalize them, and write Langfuse dataset items.
    """

    sync_started_at = time.perf_counter()
    sync_logger = bind_logger(
        logger,
        hf_dataset=hf_dataset,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        revision=revision,
        target_dataset=langfuse_dataset,
    )
    sync_logger.info(
        "sync_hf_started",
    )
    tasks = fetch_hf_localization_dataset(
        hf_dataset,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        revision=revision,
        token=token,
    )
    sync_logger.info(
        "langfuse_dataset_payload_build_started",
        source_items=len(tasks),
    )
    items = build_langfuse_items_from_hf_tasks(
        tasks,
        dataset_name=hf_dataset,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        revision=revision,
    )
    client = get_langfuse_client()
    existing_items = _fetch_existing_langfuse_items(
        client,
        dataset_name=langfuse_dataset,
    )
    existing_identities = _index_existing_item_identities(existing_items)
    original_existing_count = len(existing_identities)

    compare_started_at = time.perf_counter()
    sync_logger.info(
        "langfuse_dataset_comparison_started",
        source_items=len(items),
        existing_identities=original_existing_count,
    )
    missing_items: list[dict[str, object]] = []
    skipped_items = 0
    total_items = len(items)
    for index, item in enumerate(items, start=1):
        stable_identity = str(item["id"])
        if stable_identity in existing_identities:
            skipped_items += 1
        else:
            missing_items.append(item)
        if index % SYNC_PROGRESS_INTERVAL == 0 or index == total_items:
            logger.info(
                "langfuse_dataset_comparison_progress",
                processed=index,
                total=total_items,
                missing=len(missing_items),
                skipped=skipped_items,
            )
    sync_logger.info(
        "langfuse_dataset_comparison_completed",
        source_items=total_items,
        missing=len(missing_items),
        skipped=skipped_items,
        elapsed=_elapsed(compare_started_at),
    )

    created_items = 0
    failed_items = 0
    create_started_at = time.perf_counter()
    missing_total = len(missing_items)
    sync_logger.info(
        "langfuse_dataset_create_missing_started",
        missing=missing_total,
    )
    for index, item in enumerate(missing_items, start=1):
        stable_identity = str(item["id"])
        # Langfuse Python exposes create/list helpers, not a reliable item upsert
        # method. We store and compare our canonical LCA identity so repeated HF
        # syncs create only missing dataset items.
        try:
            _create_langfuse_dataset_item(
                client,
                dataset_name=langfuse_dataset,
                item=item,
            )
        except Exception as exc:
            failed_items += 1
            sync_logger.exception(
                "langfuse_dataset_item_create_failed",
                index=index,
                item_id=stable_identity,
                created=created_items,
                failed=failed_items,
            )
            raise HFDatasetLoadError(
                f"Failed creating Langfuse dataset item id={stable_identity}: {exc}",
                category="langfuse-client",
            ) from exc
        existing_identities.add(stable_identity)
        created_items += 1
        if index % SYNC_PROGRESS_INTERVAL == 0 or index == missing_total:
            sync_logger.info(
                "langfuse_dataset_create_missing_progress",
                created=index,
                total=missing_total,
                failed=failed_items,
            )
    sync_logger.info(
        "langfuse_dataset_create_missing_completed",
        created=created_items,
        failed=failed_items,
        elapsed=_elapsed(create_started_at),
    )
    sync_logger.info(
        "sync_hf_completed",
        source_items=len(items),
        existing=original_existing_count,
        skipped=skipped_items,
        created=created_items,
        failed=failed_items,
        total_elapsed=_elapsed(sync_started_at),
    )
    return HFDatasetSyncResult(
        items=items,
        source_items=len(items),
        existing_items=original_existing_count,
        created_items=created_items,
        skipped_items=skipped_items,
        failed_items=failed_items,
    )
