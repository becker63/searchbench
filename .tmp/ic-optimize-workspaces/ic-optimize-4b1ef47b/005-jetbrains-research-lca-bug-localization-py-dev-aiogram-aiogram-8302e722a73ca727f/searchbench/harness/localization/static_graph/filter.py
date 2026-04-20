from pathlib import Path

_SUPPORTED_SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hpp",
    ".hh",
    ".cs",
    ".rb",
    ".php",
    ".scala",
    ".swift",
    ".lua",
    ".m",
    ".mm",
    ".sh",
    ".bash",
    ".zsh",
}

_BLOCKED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".bin",
    ".so",
    ".dylib",
    ".dll",
    ".o",
    ".a",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".jar",
    ".class",
    ".ipynb",
    ".md",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".jsonl",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".lock",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".sql",
    ".jinja",
    ".j2",
    ".tmpl",
    ".tpl",
}

_BLOCKED_NAMES = {
    "license",
    "version",
    "bump",
}


def _should_skip_path(path: Path) -> bool:
    if any(
        segment in path.parts
        for segment in (
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
        )
    ):
        return True

    name = path.name.lower()
    suffix = path.suffix.lower()

    if name.startswith(".") and suffix not in _SUPPORTED_SOURCE_SUFFIXES:
        return True
    if name in _BLOCKED_NAMES:
        return True
    if suffix in _BLOCKED_SUFFIXES:
        return True
    if not is_supported_source_file(path):
        return True

    return False


def is_supported_source_file(path: Path) -> bool:
    return path.suffix.lower() in _SUPPORTED_SOURCE_SUFFIXES


def _is_probably_text(path: Path, sample_size: int = 4096) -> bool:
    try:
        data = path.read_bytes()[:sample_size]
    except OSError:
        return False

    if b"\x00" in data:
        return False

    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


__all__ = ["_should_skip_path", "_is_probably_text", "is_supported_source_file"]
