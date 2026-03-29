from __future__ import annotations

from typing import Mapping

from harness.observability import runner_integration


class _FakeResult:
    def __init__(self):
        self.id = "rid"
        self.items = [{"input": {"repo": "r"}, "output": {"ok": True}, "trace_id": "tid"}]


def test_run_hosted_dataset_experiment_uses_get_dataset(monkeypatch):
    calls: dict[str, int] = {"get_dataset": 0, "run_experiment": 0}

    class FakeDataset:
        def run_experiment(self, **kwargs):
            calls["run_experiment"] += 1
            assert kwargs["name"] == "exp"
            assert callable(kwargs["task"])
            return _FakeResult()

    class FakeClient:
        def get_dataset(self, name: str, version: str | None = None):
            calls["get_dataset"] += 1
            assert name == "ds"
            assert version == "v1"
            return FakeDataset()

    monkeypatch.setattr(runner_integration, "get_langfuse_client", lambda: FakeClient())
    result = runner_integration.run_hosted_dataset_experiment(
        dataset_name="ds",
        dataset_version="v1",
        experiment_name="exp",
        task_fn=lambda item, trace=None: {"ok": True},
    )
    assert result.get("run_id") == "rid"
    assert calls["get_dataset"] == 1 and calls["run_experiment"] == 1


def test_run_local_dataset_experiment_calls_client(monkeypatch):
    calls: dict[str, int] = {"run_experiment": 0}

    class FakeClient:
        def run_experiment(self, **kwargs):
            calls["run_experiment"] += 1
            assert kwargs["name"] == "exp_local"
            assert isinstance(kwargs["data"], list)
            return _FakeResult()

    monkeypatch.setattr(runner_integration, "get_langfuse_client", lambda: FakeClient())
    data: list[Mapping[str, object]] = [{"repo": "r"}]
    result = runner_integration.run_local_dataset_experiment(
        data=data,
        experiment_name="exp_local",
        task_fn=lambda item, trace=None: {"ok": True},
    )
    assert result.get("run_id") == "rid"
    assert calls["run_experiment"] == 1
