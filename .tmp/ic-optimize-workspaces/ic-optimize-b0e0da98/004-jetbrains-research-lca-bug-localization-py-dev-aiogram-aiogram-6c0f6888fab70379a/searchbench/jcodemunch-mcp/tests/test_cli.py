"""CLI behavior tests."""

import pytest

from jcodemunch_mcp.server import main


def test_config_init_creates_template(monkeypatch, tmp_path, capsys):
    """config --init should create a template config.jsonc file."""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.jsonc"

    monkeypatch.setenv("CODE_INDEX_PATH", str(config_dir))
    monkeypatch.setattr("sys.argv", ["jcodemunch-mcp", "config", "--init"])

    main(["config", "--init"])

    assert config_path.exists()

    # Should be valid JSONC (parseable after stripping comments)
    content = config_path.read_text()
    assert "// jcodemunch-mcp configuration" in content
    from jcodemunch_mcp.config import _strip_jsonc
    import json
    stripped = _strip_jsonc(content)
    parsed = json.loads(stripped)  # Should not raise
    assert "languages" in parsed
    assert "disabled_tools" in parsed


def test_config_init_refuses_overwrite(monkeypatch, tmp_path, capsys):
    """config --init should refuse to overwrite existing config."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.jsonc"
    config_path.write_text('{"already": "exists"}')

    monkeypatch.setenv("CODE_INDEX_PATH", str(config_dir))
    monkeypatch.setattr("sys.argv", ["jcodemunch-mcp", "config", "--init"])

    main(["config", "--init"])

    out = capsys.readouterr().out
    # Should refuse to overwrite
    assert "already exists" in out.lower() or "exists" in out.lower()


def test_main_help_exits_without_starting_server(capsys):
    """`--help` should print usage and exit cleanly."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "jcodemunch-mcp" in out
    assert "jCodeMunch MCP server" in out


def test_main_version_exits_with_version(capsys):
    """`--version` should print package version and exit cleanly."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("jcodemunch-mcp ")
