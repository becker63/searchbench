from __future__ import annotations

import json
import os
from pathlib import Path

from harness.loop import run_loop
from harness.utils.debug import print_final_summary


def main() -> None:
    """
    Simple entrypoint for running the harness loop.
    """

    # Minimal default task (can be edited by user)
    repo = os.environ.get("TEST_REPO_SMALL")
    if not repo:
        repo = str(Path(".").resolve())
    task = {
        "symbol": "expand_node",
        "repo": repo,
    }

    iterations = 5

    print("[RUN] Starting harness loop...")
    print(f"[RUN] Task: {task}")
    print(f"[RUN] Iterations: {iterations}")

    history: list[dict] = []
    try:
        history = run_loop(task, iterations=iterations)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C detected")

    print("\n=== FINAL SUMMARY ===")
    print_final_summary(history)
    print("\n[RESULT] History:")
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
