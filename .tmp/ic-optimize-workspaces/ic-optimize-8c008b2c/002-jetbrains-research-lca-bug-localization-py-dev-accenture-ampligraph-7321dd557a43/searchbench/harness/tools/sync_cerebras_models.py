"""
One-off helper to sync Cerebras pricing definitions to Langfuse Cloud Models API.
Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and optional LANGFUSE_BASE_URL) to be set.
"""

from __future__ import annotations

from harness.telemetry.tracing.cerebras_pricing import sync_cerebras_models


def main() -> None:
    sync_cerebras_models()
    print("Synced Cerebras models to Langfuse Cloud.")


if __name__ == "__main__":
    main()
