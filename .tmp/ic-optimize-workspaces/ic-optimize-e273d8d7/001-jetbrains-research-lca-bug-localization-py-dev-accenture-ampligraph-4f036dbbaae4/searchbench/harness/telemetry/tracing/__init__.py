from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Mapping, Optional, ContextManager, Iterator

from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan, propagate_attributes
from langfuse.openai import OpenAI as LangfuseOpenAI  # pyright: ignore[reportPrivateImportUsage]
from pydantic import BaseModel, ConfigDict

from harness.utils.env import get_langfuse_env

_client: Optional[Langfuse] = None

_PROPAGATION_MAX_LEN = 200
_LOGGER = logging.getLogger(__name__)
_STRICT_DEBUG = bool(os.getenv("LANGFUSE_STRICT_DEBUG"))
_PK_PREFIX_LEN = 12
_FORBIDDEN_PROPAGATED_METADATA_KEYS = {
    "repo",
    "repopath",
    "task",
    "taskidentity",
    "writersessionid",
    "optimizerunid",
    "scorecontext",
    "tasksummary",
    "reducersummary",
    "iterationdiagnostics",
}


class _OtelShutdownNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Failed to export span batch due to timeout, max retries or shutdown" not in record.getMessage()


_OTEL_EXPORTER_LOGGER = logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter")
_OTEL_NOISE_FILTER = _OtelShutdownNoiseFilter()
if not any(isinstance(filter_obj, _OtelShutdownNoiseFilter) for filter_obj in _OTEL_EXPORTER_LOGGER.filters):
    _OTEL_EXPORTER_LOGGER.addFilter(_OTEL_NOISE_FILTER)


class ObservationMetadata(BaseModel):
    """Typed metadata envelope for observations."""

    model_config = ConfigDict(extra="forbid")

    data: Mapping[str, object] | None = None

    def to_langfuse(self) -> dict[str, object] | None:
        if self.data is None:
            return None
        return serialize_observation_metadata(self.data)


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


def _is_valid_propagated_key(key: str) -> bool:
    return bool(key) and key.isalnum()


def serialize_propagated_metadata(metadata: Mapping[str, object] | None) -> dict[str, str] | None:
    """Serialize Langfuse propagated metadata.

    Langfuse v4 propagated metadata is for attributes applied across all
    observations in a context. The docs constrain it to string values of at
    most 200 characters and alphanumeric metadata keys. Execution diagnostics
    belong on observation-local metadata instead.
    """
    if metadata is None:
        return None
    serialized: dict[str, str] = {}
    for key, value in metadata.items():
        key_str = str(key)
        normalized_key = "".join(ch for ch in key_str.lower() if ch.isalnum())
        if normalized_key in _FORBIDDEN_PROPAGATED_METADATA_KEYS:
            if _STRICT_DEBUG:
                raise ValueError(f"Propagated metadata key '{key_str}' is forbidden in strict mode")
            _LOGGER.warning("Dropping execution diagnostic propagated metadata key '%s'", key_str)
            continue
        if not _is_valid_propagated_key(key_str):
            if _STRICT_DEBUG:
                raise ValueError(f"Propagated metadata key '{key_str}' is invalid in strict mode")
            _LOGGER.warning("Dropping invalid propagated metadata key '%s'", key_str)
            continue
        text = "" if value is None else str(value)
        if len(text) > _PROPAGATION_MAX_LEN:
            if _STRICT_DEBUG:
                raise ValueError(f"Propagated metadata for key '{key_str}' exceeds {_PROPAGATION_MAX_LEN} chars in strict mode")
            _LOGGER.warning(
                "Dropping propagated metadata key '%s': value exceeds %s chars",
                key_str,
                _PROPAGATION_MAX_LEN,
            )
            continue
        serialized[key_str] = text
    return serialized or None


def _json_safe(value: object, *, limit: int = 4000) -> object:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, limit=limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item, limit=limit) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item, limit=limit) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return f"{value[: limit - 3]}..."
        return value
    return str(value)


def serialize_observation_metadata(metadata: Mapping[str, object] | None) -> dict[str, object] | None:
    """Serialize observation-local metadata while preserving JSON structure."""
    if metadata is None:
        return None
    return {str(key): _json_safe(value) for key, value in metadata.items()}

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
    try:
        return LangfuseOpenAI(base_url=base_url, api_key=api_key)
    except Exception as exc:  # pragma: no cover - fallback path
        _LOGGER.warning("Langfuse OpenAI client unavailable, falling back to plain OpenAI: %s", exc)
        try:
            from openai import OpenAI
        except Exception:
            raise
        return OpenAI(base_url=base_url, api_key=api_key)


def propagate_context(
    *,
    trace_name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: Mapping[str, object] | None = None,
    version: str | None = None,
) -> ContextManager[Any]:
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
    coerced_metadata = serialize_propagated_metadata(metadata)
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
    usage_details: UsageDetails | None = None,
) -> Iterator[LangfuseSpan | LangfuseGeneration]:
    """
    Start a root observation as the current observation.
    """
    meta = ObservationMetadata(data=serialize_observation_metadata(metadata)) if metadata else None
    envelope = ObservationEnvelope(
        name=name,
        as_type=as_type,
        input=input,
        model=model,
        metadata=meta,
        usage_details=usage_details,
    )
    try:
        cm = get_langfuse_client().start_as_current_observation(**envelope.to_start_kwargs())
    except AttributeError as exc:
        raise RuntimeError("Langfuse client missing start_as_current_observation") from exc
    with cm as observation:
        yield observation


@contextmanager
def start_child_observation(
    parent: LangfuseSpan | LangfuseGeneration,
    name: str,
    *,
    as_type: str = "span",
    input: object | None = None,
    model: str | None = None,
    metadata: Mapping[str, object] | None = None,
    usage_details: UsageDetails | None = None,
) -> Iterator[LangfuseSpan | LangfuseGeneration]:
    """
    Start a child observation from a parent observation.
    """
    effective_metadata: dict[str, object] = dict(metadata) if metadata else {}
    merged_metadata: Mapping[str, object] | None = effective_metadata if effective_metadata else metadata
    meta = ObservationMetadata(data=serialize_observation_metadata(merged_metadata)) if merged_metadata else None
    envelope = ObservationEnvelope(
        name=name,
        as_type=as_type,
        input=input,
        model=model,
        metadata=meta,
        usage_details=usage_details,
    )
    try:
        cm = parent.start_as_current_observation(**envelope.to_start_kwargs())
    except AttributeError as exc:
        raise RuntimeError("Parent observation missing start_as_current_observation") from exc
    with cm as observation:
        yield observation


@contextmanager
def start_observation(
    name: str,
    *,
    parent: LangfuseSpan | LangfuseGeneration | None = None,
    as_type: str = "span",
    input: object | None = None,
    model: str | None = None,
    metadata: Mapping[str, object] | None = None,
    usage_details: UsageDetails | None = None,
) -> Iterator[LangfuseSpan | LangfuseGeneration]:
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
            usage_details=usage_details,
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
            usage_details=usage_details,
        ) as observation:
            yield observation


def flush_langfuse() -> None:
    client = get_langfuse_client()
    try:
        client.flush()
    except AttributeError:
        msg = "Langfuse client missing flush()"
        if _STRICT_DEBUG:
            raise RuntimeError(msg)
        _LOGGER.warning(msg)
    except Exception as exc:  # pragma: no cover - best effort
        if _STRICT_DEBUG:
            raise
        _LOGGER.warning("Langfuse flush failed: %s", exc)
    try:
        client.shutdown()
    except AttributeError:
        msg = "Langfuse client missing shutdown()"
        if _STRICT_DEBUG:
            raise RuntimeError(msg)
        _LOGGER.warning(msg)
    except Exception as exc:  # pragma: no cover - best effort
        if _STRICT_DEBUG:
            raise
        _LOGGER.warning("Langfuse shutdown failed: %s", exc)


def ensure_langfuse_auth() -> None:
    """
    Perform an explicit Langfuse auth/base-url check.
    Raises in strict mode to surface misconfigurations instead of continuing silently.
    """
    client = get_langfuse_client()
    public_prefix: str | None = None
    host: str | None = None
    try:
        env = get_langfuse_env()
        public = env.get("public_key")
        if isinstance(public, str):
            public_prefix = public[:_PK_PREFIX_LEN]
        host_val = env.get("base_url")
        host = host_val if isinstance(host_val, str) and host_val else None
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Langfuse environment metadata inspection failed: %s", exc)
    try:
        ok = bool(client.auth_check())
    except AttributeError:
        msg = "Langfuse client does not expose auth_check; cannot verify credentials"
        if _STRICT_DEBUG:
            raise RuntimeError(msg)
        _LOGGER.warning(msg)
        return
    except Exception as exc:  # pragma: no cover
        detail = f"Langfuse auth_check raised: {exc}"
        if _STRICT_DEBUG:
            raise RuntimeError(detail) from exc
        _LOGGER.warning(detail)
        return
    if not ok:
        detail = (
            "Langfuse auth_check failed; verify LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            f"and LANGFUSE_BASE_URL (host={host or 'unset'}, pk_prefix={public_prefix or 'unset'})"
        )
        if _STRICT_DEBUG:
            raise RuntimeError(detail)
        _LOGGER.warning(detail)


__all__ = [
    "get_langfuse_client",
    "is_langfuse_enabled",
    "get_tracing_openai_client",
    "propagate_context",
    "start_observation",
    "start_root_observation",
    "start_child_observation",
    "flush_langfuse",
    "ensure_langfuse_auth",
    "serialize_observation_metadata",
    "serialize_propagated_metadata",
]
