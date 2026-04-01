from __future__ import annotations

from typing import Any, Mapping, Optional, cast

from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI  # pyright: ignore[reportPrivateImportUsage]

from ..utils.env import get_langfuse_env

_client: Optional[Langfuse] = None


def get_langfuse_client() -> Optional[Langfuse]:
    global _client  # noqa: PLW0603
    env = get_langfuse_env()
    public, secret = env.get("public_key"), env.get("secret_key")
    if not isinstance(public, str) or not isinstance(secret, str) or not public or not secret:
        raise RuntimeError("Langfuse credentials are required (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)")
    if _client is not None:
        return _client
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
        raise
    return _client


def is_langfuse_enabled() -> bool:
    return get_langfuse_client() is not None


def get_tracing_openai_client(base_url: str, api_key: str) -> Any:
    """
    Require Langfuse's OpenAI integration for tracing.
    """
    return LangfuseOpenAI(base_url=base_url, api_key=api_key)


def start_trace(name: str, metadata: Mapping[str, object] | None = None, input: object | None = None) -> Any | None:
    client = get_langfuse_client()
    client_any = cast(Any, client)
    return client_any.trace(name=name, metadata=dict(metadata) if metadata else None, input=input)


def start_span(parent: Any | None, name: str, metadata: Mapping[str, object] | None = None, input: object | None = None) -> Any | None:
    if parent is None:
        return start_trace(name=name, metadata=metadata, input=input)
    if hasattr(parent, "span"):
        return parent.span(name=name, metadata=dict(metadata) if metadata else None, input=input)
    raise RuntimeError("Parent trace/span missing 'span' method")


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse client unavailable for flush")
    client.flush()


__all__ = [
    "get_langfuse_client",
    "is_langfuse_enabled",
    "get_tracing_openai_client",
    "start_trace",
    "start_span",
    "flush_langfuse",
]
