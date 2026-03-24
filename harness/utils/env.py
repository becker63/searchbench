from __future__ import annotations


from dotenv import load_dotenv

_loaded = False


def load_env() -> None:
    global _loaded  # noqa: PLW0603
    if not _loaded:
        load_dotenv()
        _loaded = True
