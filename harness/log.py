from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

import structlog
from structlog.stdlib import BoundLogger

if TYPE_CHECKING:
    from harness.localization.models import LCATask, LCATaskIdentity


_ISSUE_RE = re.compile(r"/(?:issues|pull)/(?P<number>\d+)")


def get_logger(name: str | None = None) -> BoundLogger:
    return structlog.stdlib.get_logger(name)


def bind_logger(logger: BoundLogger, **context: object) -> BoundLogger:
    clean = {key: value for key, value in context.items() if value is not None}
    return logger.bind(**clean) if clean else logger


def short_task_label(
    task: "LCATask | LCATaskIdentity | str",
    *,
    ordinal: int | None = None,
    total: int | None = None,
) -> str:
    from harness.localization.models import LCATask, LCATaskIdentity

    repo_slug: str | None = None
    issue_ref: str | None = None
    fallback_seed: str

    if isinstance(task, LCATask):
        identity = task.identity
        repo_slug = f"{identity.repo_owner}/{identity.repo_name}"
        issue_ref = _issue_ref(identity.issue_url or identity.pull_url)
        fallback_seed = task.task_id
    elif isinstance(task, LCATaskIdentity):
        repo_slug = f"{task.repo_owner}/{task.repo_name}"
        issue_ref = _issue_ref(task.issue_url or task.pull_url)
        fallback_seed = task.task_id()
    else:
        fallback_seed = str(task)
        repo_slug, issue_ref = _parts_from_task_id(fallback_seed)

    base = repo_slug or "task"
    ref = issue_ref or _stable_suffix(fallback_seed)
    label = f"{base}{ref}"
    if ordinal is not None and total is not None:
        return f"{ordinal}/{total} {label}"
    if ordinal is not None:
        return f"{ordinal} {label}"
    return label


def tail_text(text: str | None, *, max_chars: int = 400) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:]


def _issue_ref(url: str | None) -> str | None:
    if not url:
        return None
    match = _ISSUE_RE.search(url)
    if not match:
        return None
    return f"#{match.group('number')}"


def _parts_from_task_id(task_id: str) -> tuple[str | None, str | None]:
    parts = task_id.split(":")
    repo_part = next((part for part in parts if "/" in part and "@" in part), None)
    repo_slug = repo_part.split("@", 1)[0] if repo_part else None
    url_part = next((part for part in parts if "http" in part), None)
    return repo_slug, _issue_ref(url_part)


def _stable_suffix(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return f":{digest}"


__all__ = ["bind_logger", "get_logger", "short_task_label", "tail_text"]
