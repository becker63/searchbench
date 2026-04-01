from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Optional

from langfuse import Langfuse, propagate_attributes
from langfuse.openai import OpenAI as LangfuseOpenAI  # pyright: ignore[reportPrivateImportUsage]

from ..utils.env import get_langfuse_env

_client: Optional[Langfuse] = None


def _coerce_metadata(metadata: Mapping[str, object] | None) -> dict[str, str] | None:
    if metadata is None:
        return None
    coerced: dict[str, str] = {}
    for key, value in metadata.items():
        coerced[str(key)] = "" if value is None else str(value)
    return coerced


def _coerce_tags(tags: list[str] | None) -> list[str] | None:
    if tags is None:
        return None
    return [tag for tag in tags if isinstance(tag, str) and tag]


def _build_client() -> Langfuse:
    env = get_langfuse_env()
    public, secret = env.get("public_key"), env.get("secret_key")
    if not isinstance(public, str) or not isinstance(secret, str) or not public or not secret:
        raise RuntimeError("Langfuse credentials are required (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)")
    host = env.get("base_url")
    tracing_env = env.get("tracing_environment")
    release = env.get("release")
    debug = bool(env.get("debug"))
    return Langfuse(
        public_key=public,
        secret_key=secret,
        host=host if isinstance(host, str) and host else None,
        environment=tracing_env if isinstance(tracing_env, str) and tracing_env else None,
        release=release if isinstance(release, str) and release else None,
        debug=debug,
        # Keep default span filter (no infra spans) by not overriding should_export_span.
    )


def get_langfuse_client() -> Langfuse:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = _build_client()
    return _client


def is_langfuse_enabled() -> bool:
    try:
        get_langfuse_client()
        return True
    except Exception:
        return False


def get_tracing_openai_client(base_url: str, api_key: str) -> Any:
    """
    Require Langfuse's OpenAI integration for tracing.
    """
    return LangfuseOpenAI(base_url=base_url, api_key=api_key)


def propagate_context(
    *,
    trace_name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: Mapping[str, object] | None = None,
    version: str | None = None,
):
    """
    Apply correlating attributes to all observations created in the current context.
    """
    kwargs: dict[str, Any] = {}
    if trace_name:
        kwargs["trace_name"] = trace_name
    if session_id:
        kwargs["session_id"] = session_id
    if user_id:
        kwargs["user_id"] = user_id
    coerced_tags = _coerce_tags(tags)
    if coerced_tags:
        kwargs["tags"] = coerced_tags
    coerced_metadata = _coerce_metadata(metadata)
    if coerced_metadata:
        kwargs["metadata"] = coerced_metadata
    if version:
        kwargs["version"] = version
    return propagate_attributes(**kwargs)


@contextmanager
def start_root_observation(
    name: str,
    *,
    as_type: str = "span",
    input: object | None = None,
    model: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Any:
    """
    Start a root observation as the current observation.
    """
    client = get_langfuse_client()
    with client.start_as_current_observation(
        as_type=as_type,
        name=name,
        input=input,
        model=model,
        metadata=_coerce_metadata(metadata),
    ) as observation:
        yield observation


@contextmanager
def start_child_observation(
    parent: Any,
    name: str,
    *,
    as_type: str = "span",
    input: object | None = None,
    model: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Any:
    """
    Start a child observation from a parent observation.
    """
    if parent is None or not hasattr(parent, "start_observation"):
        raise RuntimeError("Parent observation is required to start a child observation")
    with parent.start_observation(
        name=name,
        as_type=as_type,
        input=input,
        model=model,
        metadata=_coerce_metadata(metadata),
    ) as observation:
        yield observation


@contextmanager
def start_observation(
    name: str,
    *,
    parent: Any | None = None,
    as_type: str = "span",
    input: object | None = None,
    model: str | None = None,
    metadata: Mapping[str, object] | None = None,
):
    """
    Convenience wrapper to start a root or child observation based on parent presence.
    """
    if parent is None:
        with start_root_observation(
            name,
            as_type=as_type,
            input=input,
            model=model,
            metadata=metadata,
        ) as observation:
            yield observation
    else:
        with start_child_observation(
            parent,
            name,
            as_type=as_type,
            input=input,
            model=model,
            metadata=metadata,
        ) as observation:
            yield observation


def flush_langfuse() -> None:
    client = get_langfuse_client()
    # Ensure buffered events are sent; shutdown if supported by the SDK.
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown()


__all__ = [
    "get_langfuse_client",
    "is_langfuse_enabled",
    "get_tracing_openai_client",
    "propagate_context",
    "start_observation",
    "start_root_observation",
    "start_child_observation",
    "flush_langfuse",
]
