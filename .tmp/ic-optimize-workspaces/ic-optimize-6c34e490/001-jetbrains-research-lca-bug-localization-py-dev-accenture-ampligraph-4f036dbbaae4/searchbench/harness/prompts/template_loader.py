from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, select_autoescape

from harness.utils.repo_root import find_repo_root


def _resolve_prompts_dir() -> Path:
    root = find_repo_root()
    candidates = [
        root / "prompts",
        root / "harness" / "prompts",
        Path(__file__).resolve().parent,
    ]
    prompts_dir = next((p for p in candidates if p.exists()), None)
    if prompts_dir is None:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(f"Prompts directory missing (checked: {checked})")
    return prompts_dir


def render_prompt_template(template_name: str, context: Mapping[str, Any]) -> str:
    """Render a Jinja2 template from the harness.prompts package."""

    env = Environment(
        loader=FileSystemLoader(_resolve_prompts_dir()),
        autoescape=select_autoescape(enabled_extensions=(), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    try:
        template = env.get_template(template_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Prompt template '{template_name}' not found") from exc
    try:
        return template.render(**context)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to render template '{template_name}'") from exc


__all__ = ["render_prompt_template"]
