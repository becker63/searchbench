from __future__ import annotations

import logging
from typing import Any, Iterable, List, Mapping, Sequence

from harness.localization.models import LCATask, normalize_lca_task

logger = logging.getLogger(__name__)


class HFDatasetLoadError(RuntimeError):
    """
    Error raised when HF dataset loading fails.
    `category` is used to surface failure class in CLI/telemetry.
    """

    def __init__(self, message: str, category: str = "unknown"):
        super().__init__(message)
        self.category = category


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
    validated: List[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise HFDatasetLoadError("HF row is not a mapping", category="schema")
        required = ["repo_owner", "repo_name", "base_sha", "changed_files"]
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
