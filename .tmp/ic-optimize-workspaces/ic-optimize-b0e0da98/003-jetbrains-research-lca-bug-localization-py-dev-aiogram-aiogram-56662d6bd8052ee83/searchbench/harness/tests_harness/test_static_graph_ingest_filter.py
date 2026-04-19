from __future__ import annotations

from pathlib import Path

from harness.localization.static_graph import ingest as ingest_mod
from harness.localization.static_graph.filter import (
    _should_skip_path,
    is_supported_source_file,
)


def test_supported_source_policy_accepts_real_source_files():
    accepted = [
        Path("pkg/app.py"),
        Path("web/app.js"),
        Path("web/app.ts"),
        Path("src/Main.java"),
        Path("src/Main.kt"),
        Path("cmd/main.go"),
        Path("src/lib.rs"),
        Path("src/main.cpp"),
        Path("src/Program.cs"),
        Path("scripts/run.sh"),
    ]

    assert all(is_supported_source_file(path) for path in accepted)
    assert all(not _should_skip_path(path) for path in accepted)


def test_supported_source_policy_skips_non_code_text_and_config_files():
    skipped = [
        Path("README.md"),
        Path("notes.txt"),
        Path("config.yaml"),
        Path("pyproject.toml"),
        Path("package.json"),
        Path("setup.cfg"),
        Path("settings.ini"),
    ]

    assert all(not is_supported_source_file(path) for path in skipped)
    assert all(_should_skip_path(path) for path in skipped)


def test_supported_source_policy_skips_extensionless_and_templates():
    skipped = [
        Path("Dockerfile"),
        Path("Makefile"),
        Path("templates/page.html"),
        Path("templates/page.jinja"),
        Path("templates/page.tmpl"),
        Path("notebooks/example.ipynb"),
    ]

    assert all(not is_supported_source_file(path) for path in skipped)
    assert all(_should_skip_path(path) for path in skipped)


def test_ingest_repo_only_passes_supported_source_files_to_extractor(
    monkeypatch, tmp_path: Path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    files = {
        "src/app.py": "def run():\n    return 1\n",
        "src/app.js": "function run() { return 1; }\n",
        "README.md": "# docs\n",
        "config.yaml": "key: value\n",
        "pyproject.toml": "[project]\nname = 'x'\n",
        "package.json": '{"scripts": {}}\n',
        "template.html": "<div>{{ value }}</div>\n",
        "Dockerfile": "FROM python:3.13\n",
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    extracted_paths: list[Path] = []

    def fake_extract_file(path: str):
        extracted_paths.append(Path(path))
        return {"functions": [{"name": "run"}], "call_graph": {"calls": {}}}

    monkeypatch.setattr(ingest_mod, "extract_file", fake_extract_file)
    monkeypatch.setattr(ingest_mod, "get_imports", lambda path: [])
    monkeypatch.setattr(ingest_mod, "ModuleInfo", None)
    monkeypatch.setattr(ingest_mod, "FunctionInfo", None)
    monkeypatch.setattr(ingest_mod, "ImportInfo", None)

    tree = ingest_mod.ingest_repo(repo)

    assert {path.relative_to(repo).as_posix() for path in extracted_paths} == {
        "src/app.py",
        "src/app.js",
    }
    graph_paths = {file.path for file in tree.files}
    assert graph_paths == {"src/app.py", "src/app.js"}
