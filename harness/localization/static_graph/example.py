from __future__ import annotations

from ..scoring_models import SCORE_REGISTRY, ScoreConfig, ScoreContext, ScoreEngine


def build_example_context() -> ScoreContext:
    return ScoreContext(
        predicted_files=("src/repair.py", "src/parser.py"),
        gold_files=("src/repair.py",),
        gold_min_hops=0,
        issue_min_hops=2,
        per_prediction_hops={"src/repair.py": 0, "src/parser.py": 2},
        previous_total_token_count=8000,
        resolved_prediction_nodes={
            "src/repair.py": ("src/repair.py",),
            "src/parser.py": ("src/parser.py",),
        },
        resolved_gold_nodes={"src/repair.py": ("src/repair.py",)},
        issue_anchor_nodes=("src/repair.py",),
    )


def main() -> None:
    engine = ScoreEngine(SCORE_REGISTRY, ScoreConfig.default())
    bundle = engine.evaluate(build_example_context())

    print("COMPOSED MACHINE SCORE")
    print(f"  mode:    {bundle.compose_mode.value}")
    print(f"  score:   {bundle.composed_score}")
    print(f"  weights: {dict(bundle.compose_weights)}")

    print("\nVISIBLE SCORES FOR AGENT")
    for name, result in bundle.results.items():
        if result.visible_to_agent:
            print(
                f"  {name:16} "
                f"value={result.value!r:<12} "
                f"available={result.available!s:<5} "
                f"metadata={dict(result.metadata)}"
            )


if __name__ == "__main__":
    main()
