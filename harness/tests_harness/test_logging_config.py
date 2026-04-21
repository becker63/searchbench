from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from harness.log import get_logger, short_task_label
from harness.logging_config import configure_logging
from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity


def _json_events(stderr: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in stderr.splitlines()
        if line.strip().startswith("{")
    ]


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    configure_logging(
        level=logging.INFO,
        renderer="console",
        force=True,
        cache_logger_on_first_use=False,
    )


def test_configure_logging_emits_structured_json_for_structlog_and_stdlib(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(
        level=logging.INFO,
        renderer="json",
        force=True,
        cache_logger_on_first_use=False,
    )

    get_logger("harness.test").bind(optimize_run_id="run-1", iteration=2).info(
        "optimize_iteration_completed",
        status="success",
    )
    logging.getLogger("plain.test").warning("plain_warning")

    events = _json_events(capsys.readouterr().err)
    assert len(events) == 2

    structlog_event, stdlib_event = events
    assert structlog_event["event"] == "optimize_iteration_completed"
    assert structlog_event["optimize_run_id"] == "run-1"
    assert structlog_event["iteration"] == 2
    assert structlog_event["status"] == "success"
    assert structlog_event["logger"] == "harness.test"
    assert structlog_event["level"] == "info"

    assert stdlib_event["event"] == "plain_warning"
    assert stdlib_event["logger"] == "plain.test"
    assert stdlib_event["level"] == "warning"


def test_short_task_label_prefers_repo_and_issue_number() -> None:
    task = LCATask(
        identity=LCATaskIdentity(
            dataset_name="lca",
            dataset_config="py",
            dataset_split="dev",
            repo_owner="octo",
            repo_name="widgets",
            base_sha="abc",
            issue_url="https://github.com/octo/widgets/issues/123",
        ),
        context=LCAContext(issue_title="bug", issue_body="details"),
        gold=LCAGold(changed_files=["a.py"]),
        repo="repo",
    )

    assert short_task_label(task, ordinal=2, total=7) == "2/7 octo/widgets#123"
