from __future__ import annotations

from typing import Any, Mapping, Optional, cast

from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI

from ..utils.env import get_langfuse_env

_client: Optional[Langfuse] = None


def get_langfuse_client() -> Optional[Langfuse]:
    global _client  # noqa: PLW0603
    env = get_langfuse_env()
    if not env.get("enabled", True):
        _client = None
        return None
    if _client is not None:
        return _client
    public, secret = env.get("public_key"), env.get("secret_key")
    if not isinstance(public, str) or not isinstance(secret, str) or not public or not secret:
        return None
    base_url_val = env.get("base_url")
    env_val = env.get("environment")
    host: Optional[str] = base_url_val if isinstance(base_url_val, str) else None
    environment: Optional[str] = env_val if isinstance(env_val, str) else None
    try:
        _client = Langfuse(
            public_key=public,
            secret_key=secret,
            host=host if host else None,
            environment=environment if environment else None,
        )
    except Exception:
        _client = None
    return _client


def is_langfuse_enabled() -> bool:
    return get_langfuse_client() is not None


def get_tracing_openai_client(base_url: str, api_key: str) -> Any:
    """
    Prefer Langfuse's OpenAI integration, but fall back to vanilla client if unavailable.
    """
    try:
        return LangfuseOpenAI(base_url=base_url, api_key=api_key)
    except Exception:
        from openai import OpenAI  # local import to avoid import-time side effects

        return OpenAI(base_url=base_url, api_key=api_key)


def start_trace(name: str, metadata: Mapping[str, object] | None = None, input: object | None = None) -> Any | None:
    client = get_langfuse_client()
    if not client:
        return None
    try:
        client_any = cast(Any, client)
        return client_any.trace(name=name, metadata=dict(metadata) if metadata else None, input=input)
    except Exception:
        return None


def start_span(parent: Any | None, name: str, metadata: Mapping[str, object] | None = None, input: object | None = None) -> Any | None:
    if parent is None:
        return start_trace(name=name, metadata=metadata, input=input)
    try:
        if hasattr(parent, "span"):
            return parent.span(name=name, metadata=dict(metadata) if metadata else None, input=input)
    except Exception:
        return None
    return None


def record_score(handle: Any | None, name: str, value: float | int | bool, metadata: Mapping[str, object] | None = None) -> None:
    if handle is None:
        return
    try:
        scorer = getattr(handle, "score", None)
        if scorer:
            scorer(name=name, value=float(value), metadata=dict(metadata) if metadata else None)
    except Exception:
        return


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if not client:
        return
    try:
        client.flush()
    except Exception:
        return


__all__ = [
    "get_langfuse_client",
    "is_langfuse_enabled",
    "get_tracing_openai_client",
    "start_trace",
    "start_span",
    "record_score",
    "flush_langfuse",
]
