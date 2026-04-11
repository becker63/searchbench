import harness.entrypoints.cli as harness_run

# Expose key callables for tests; they may be monkeypatched on this shim.
cost_details_for_usage = harness_run.cost_details_for_usage
fetch_hf_localization_dataset = harness_run.fetch_hf_localization_dataset
run_hosted_localization_baseline = harness_run.run_hosted_localization_baseline
run_hosted_localization_experiment = harness_run.run_hosted_localization_experiment
_parse_args = harness_run._parse_args
_select_tasks = harness_run._select_tasks
_require_confirmation = harness_run._require_confirmation
_original_compute_projection = harness_run._compute_projection
termios = harness_run.termios
tty = harness_run.tty


def _compute_projection(selected_count: int):
    # Ensure any monkeypatched pricing helper is used by the underlying implementation.
    harness_run.cost_details_for_usage = cost_details_for_usage
    return _original_compute_projection(selected_count)


def evaluate_localization_batch(**kwargs):
    return harness_run.evaluate_localization_batch(**kwargs)


def main(argv: list[str] | None = None) -> None:
    # Sync patched symbols into the underlying harness.run module so tests can monkeypatch this shim.
    harness_run.evaluate_localization_batch = evaluate_localization_batch
    harness_run.fetch_hf_localization_dataset = fetch_hf_localization_dataset
    harness_run.run_hosted_localization_baseline = run_hosted_localization_baseline
    harness_run.run_hosted_localization_experiment = run_hosted_localization_experiment
    harness_run.cost_details_for_usage = cost_details_for_usage
    harness_run.termios = termios
    harness_run.tty = tty
    harness_run._parse_args = _parse_args
    harness_run._select_tasks = _select_tasks
    harness_run._compute_projection = _compute_projection
    harness_run._require_confirmation = _require_confirmation
    harness_run.main(argv)


if __name__ == "__main__":
    main()
