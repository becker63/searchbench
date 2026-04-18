from __future__ import annotations

import logging
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from harness.localization.models import LCATask, normalize_lca_task
from harness.telemetry.hosted.datasets import langfuse_dataset_item_from_task
from harness.telemetry.tracing import get_langfuse_client

logger = logging.getLogger(__name__)


class HFDatasetLoadError(RuntimeError):
    """
    Error raised when HF dataset loading fails.
    `category` is used to surface failure class in CLI/telemetry.
    """

    def __init__(self, message: str, category: str = "unknown"):
        super().__init__(message)
        self.category = category


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
    logger.info("Loaded %d HF LCA rows (config=%s split=%s)", len(tasks), dataset_config, dataset_split)
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
    return [
        langfuse_dataset_item_from_task(
            task,
            metadata={
                **metadata,
                "canonical_task_identity": compute_lca_task_identity(task),
                "task_id": task.task_id,
                "repo_owner": task.identity.repo_owner,
                "repo_name": task.identity.repo_name,
                "base_sha": task.identity.base_sha,
            },
        )
        | {"id": compute_lca_task_identity(task)}
        for task in tasks
    ]


def _upsert_langfuse_dataset_item(
    client: object,
    *,
    dataset_name: str,
    item: Mapping[str, object],
) -> None:
    item_id = item["id"]
    payload = {
        "dataset_name": dataset_name,
        "id": item_id,
        "input": item["input"],
        "expected_output": item["expected_output"],
        "metadata": item["metadata"],
    }
    upsert_item = getattr(client, "upsert_dataset_item", None)
    if callable(upsert_item):
        upsert_item(**payload)
        return
    raise HFDatasetLoadError(
        "Langfuse client does not expose dataset item upsert API; cannot sync HF dataset idempotently",
        category="langfuse-client",
    )


def sync_hf_localization_dataset_to_langfuse(
    *,
    hf_dataset: str,
    langfuse_dataset: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> list[dict[str, object]]:
    """
    Load HF localization rows, normalize them, and write Langfuse dataset items.
    """

    tasks = fetch_hf_localization_dataset(
        hf_dataset,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        revision=revision,
        token=token,
    )
    items = build_langfuse_items_from_hf_tasks(
        tasks,
        dataset_name=hf_dataset,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        revision=revision,
    )
    client = get_langfuse_client()
    for item in items:
        _upsert_langfuse_dataset_item(client, dataset_name=langfuse_dataset, item=item)
    return items
