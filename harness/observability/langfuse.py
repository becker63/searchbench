from __future__ import annotations

from contextlib import contextmanager
from types import TracebackType
from typing import Any, Iterator, Mapping, Optional, cast

from langfuse import Langfuse, propagate_attributes
from langfuse.openai import OpenAI as LangfuseOpenAI  # pyright: ignore[reportPrivateImportUsage]
from pydantic import BaseModel, ConfigDict

try:
    # Prefer absolute import so observability can be used as a top-level package in tests.
    from utils.env import get_langfuse_env
except ImportError:  # pragma: no cover
    from harness.utils.env import get_langfuse_env

_client: Optional[Langfuse] = None


class ObservationMetadata(BaseModel):
    """Typed metadata envelope for observations."""

    model_config = ConfigDict(extra="forbid")

    data: Mapping[str, object] | None = None

    def to_langfuse(self) -> dict[str, str] | None:
        if self.data is None:
            return None
        coerced: dict[str, str] = {}
        for key, value in self.data.items():
            coerced[str(key)] = "" if value is None else str(value)
        return coerced


class UsageDetails(BaseModel):
    """Structured usage/cost inputs for Langfuse/OpenAI-style responses."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


class ObservationEnvelope(BaseModel):
    """Structured envelope for observation creation/update."""

    model_config = ConfigDict(extra="forbid")

    name: str
    as_type: str = "span"
    input: object | None = None
    model: str | None = None
    metadata: ObservationMetadata | None = None
    usage_details: UsageDetails | None = None
    tags: list[str] | None = None
    session_id: str | None = None
    user_id: str | None = None
    version: str | None = None

    def to_start_kwargs(self) -> dict[str, object]:
        return {
            "as_type": self.as_type,
            "name": self.name,
            "input": self.input,
            "model": self.model,
            "metadata": self.metadata.to_langfuse() if self.metadata else None,
        }

    def propagate_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if self.session_id:
            kwargs["session_id"] = self.session_id
        if self.user_id:
            kwargs["user_id"] = self.user_id
        if self.tags:
            kwargs["tags"] = [t for t in self.tags if isinstance(t, str) and t]
        if self.version:
            kwargs["version"] = self.version
        return kwargs


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


def _compact_metadata_value(key: str, value: object, limit: int = 180) -> object:
    """
    Summarize oversized propagation values without losing key counts.
    """
    if key in {"selection", "projection"}:
        try:
            if isinstance(value, Mapping):
                preview = value.get("preview")
                def _trim_preview_item(item: object) -> object:
                    text = str(item)
                    return text if len(text) <= 60 else text[:60] + "..."

                preview_list = None
                if isinstance(preview, list):
                    preview_list = [_trim_preview_item(item) for item in preview[:3]]
                return {
                    "offset": value.get("offset"),
                    "limit": value.get("limit"),
                    "selected_count": value.get("selected_count"),
                    "total_available": value.get("total_available"),
                    "preview": preview_list or preview,
                }
            if isinstance(value, (list, tuple)):
                trimmed = list(value)[:3]
                return trimmed if len(str(trimmed)) <= limit else str(trimmed)[:limit] + "..."
            text = str(value)
            return text if len(text) <= limit else text[:limit] + "..."
        except Exception:
            return str(value)[:limit]
    text_val = str(value)
    if len(text_val) > limit:
        return text_val[:limit] + "..."
    return value


def _compact_metadata(meta: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if meta is None:
        return None
    compacted: dict[str, object] = {}
    for key, value in meta.items():
        compacted[key] = _compact_metadata_value(str(key), value)
    return compacted


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
    compact_meta = _compact_metadata(metadata)
    coerced_metadata = ObservationMetadata(data=compact_meta).to_langfuse() if compact_meta else None
    if coerced_metadata:
        kwargs["metadata"] = coerced_metadata
    if version:
        kwargs["version"] = version
    return propagate_attributes(**kwargs)


@contextmanager
def _observation_scope(started: Any) -> Iterator[Any]:
    """
    Normalize Langfuse start return values to a context manager.
    """
    if hasattr(started, "__enter__") and hasattr(started, "__exit__"):
        with started as observation:
            yield observation
        return

    try:
        yield started
    finally:
        end = getattr(started, "end", None)
        if callable(end):
            end()


def _safe_end_observation(
    observation: Any,
    *,
    error: object | None = None,
    metadata: Mapping[str, object] | None = None,
    output: object | None = None,
) -> None:
    """
    End an observation without assuming keyword support on .end().
    """
    if observation is None:
        return
    update = getattr(observation, "update", None)
    extra_meta: dict[str, object] = {}
    if metadata:
        extra_meta.update(metadata)
    if error is not None:
        extra_meta.setdefault("error", error)
    payload: dict[str, object] = {}
    if extra_meta:
        payload["metadata"] = extra_meta
    if output is not None:
        payload["output"] = output
    try:
        if callable(update) and payload:
            update(**payload)
    finally:
        end = getattr(observation, "end", None)
        if callable(end):
            end()


@contextmanager
def start_root_observation(
    name: str,
    *,
    as_type: str = "span",
    input: object | None = None,
    model: str | None = None,
    metadata: Mapping[str, object] | None = None,
    usage_details: UsageDetails | None = None,
) -> Any:
    """
    Start a root observation as the current observation.
    """
    client_any: Any = get_langfuse_client()
    envelope = ObservationEnvelope(
        name=name,
        as_type=as_type,
        input=input,
        model=model,
        metadata=ObservationMetadata(data=metadata) if metadata else None,
        usage_details=usage_details,
    )
    start_fn = getattr(client_any, "start_as_current_observation", None)
    if callable(start_fn):
        start_ctx = cast(Any, start_fn)(**envelope.to_start_kwargs())
        with _observation_scope(start_ctx) as observation:
            yield observation
        return

    # Fallback for stubbed clients in tests.
    class _DummyObs:
        def __init__(self):
            self.id = "trace"
            self.trace_id = "trace"

        def start_observation(self, **kwargs: object) -> Any:
            return self

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> bool:
            self.end()
            return False

        def end(self, **kwargs: object) -> None:
            return None

        def update(self, **kwargs: object) -> None:
            return None

    obs = _DummyObs()
    with _observation_scope(obs) as observation:
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
    usage_details: UsageDetails | None = None,
) -> Any:
    """
    Start a child observation from a parent observation.
    """
    if parent is None or not hasattr(parent, "start_observation"):
        raise RuntimeError("Parent observation is required to start a child observation")
    envelope = ObservationEnvelope(
        name=name,
        as_type=as_type,
        input=input,
        model=model,
        metadata=ObservationMetadata(data=metadata) if metadata else None,
        usage_details=usage_details,
    )
    started = parent.start_observation(**envelope.to_start_kwargs())
    with _observation_scope(started) as observation:
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
    usage_details: UsageDetails | None = None,
):
    """
    Convenience wrapper to start a root or child observation based on parent presence.
    """
    parent_ctx = parent if parent is not None and hasattr(parent, "start_observation") else None
    if parent_ctx is None:
        with start_root_observation(
            name,
            as_type=as_type,
            input=input,
            model=model,
            metadata=metadata,
            usage_details=usage_details,
        ) as observation:
            yield observation
    else:
        with start_child_observation(
            parent_ctx,
            name,
            as_type=as_type,
            input=input,
            model=model,
            metadata=metadata,
            usage_details=usage_details,
        ) as observation:
            yield observation


def flush_langfuse() -> None:
    client = get_langfuse_client()
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
