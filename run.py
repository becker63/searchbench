from harness.run import evaluate_localization_batch as _evaluate_localization_batch


def evaluate_localization_batch(req):
    return _evaluate_localization_batch(req)


def main(argv: list[str] | None = None) -> None:
    import harness.run as harness_run

    harness_run.evaluate_localization_batch = evaluate_localization_batch
    harness_run.main(argv)


if __name__ == "__main__":
    main()
