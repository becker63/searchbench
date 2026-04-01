from __future__ import annotations


from dotenv import load_dotenv

_loaded = False


def load_env() -> None:
    global _loaded  # noqa: PLW0603
    if not _loaded:
        load_dotenv()
        _loaded = True


def getenv(key: str, default: str | None = None) -> str | None:
    """
    Wrapper around os.environ that ensures .env is loaded once.
    """
    import os

    load_env()
    return os.environ.get(key, default)


def getenv_bool(key: str, default: bool = False) -> bool:
    val = getenv(key)
    if val is None:
        return default
    return val.lower() not in {"0", "false", "off", "no"}


def get_runner_model() -> str:
    return getenv("RUNNER_MODEL", "llama3.1-8b") or "llama3.1-8b"


def get_writer_model() -> str:
    return getenv("WRITER_MODEL", "gpt-oss-120b") or "gpt-oss-120b"


def get_cerebras_api_key() -> str | None:
    return getenv("CEREBRAS_API_KEY")


def get_langfuse_env() -> dict[str, str | bool | None]:
    """
    Centralized Langfuse environment access to avoid scattered lookups.
    """
    return {
        "public_key": getenv("LANGFUSE_PUBLIC_KEY"),
        "secret_key": getenv("LANGFUSE_SECRET_KEY"),
        "base_url": getenv("LANGFUSE_BASE_URL"),
        "environment": getenv("LANGFUSE_ENV"),
    }
