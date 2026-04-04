from localization.models import LCATaskIdentity, LCAPrediction, LCAGold
from localization.scoring import score_file_localization


def test_identity_is_deterministic():
    identity = LCATaskIdentity(
        dataset_name="lca-bug-localization",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="square",
        repo_name="okhttp",
        base_sha="abc123",
        issue_url="https://github.com/square/okhttp/issues/1",
    )
    assert (
        identity.task_id()
        == "lca-bug-localization:py:dev:square/okhttp@abc123:https://github.com/square/okhttp/issues/1"
    )


def test_file_set_scoring_paths_only():
    prediction = LCAPrediction(predicted_files=["a.py", "c.py"])
    gold = LCAGold(changed_files=["a.py", "b.py"])
    metrics = score_file_localization(prediction, gold)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["score"] == metrics["f1"]
