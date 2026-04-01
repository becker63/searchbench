from __future__ import annotations

from typing import Mapping, Any

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .repo_root import find_repo_root


def render_prompt_template(template_name: str, context: Mapping[str, Any]) -> str:
    """
    Render a Jinja2 template from the prompts/ directory.
    """
    root = find_repo_root()
    candidates = [
        root / "prompts",
        root / "harness" / "prompts",
        Path(__file__).resolve().parent.parent / "prompts",
    ]
    prompts_dir = next((p for p in candidates if p.exists()), None)
    if prompts_dir is None:
        raise RuntimeError(f"Prompts directory missing (checked: {', '.join(str(c) for c in candidates)})")
    env = Environment(
        loader=FileSystemLoader(prompts_dir),
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
