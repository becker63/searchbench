from __future__ import annotations

from pathlib import Path
import sys
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.tools.loop_viz import render_state_machines


def test_mermaid_render_to_stdout(capsys):
    render_state_machines(fmt="mermaid", machine="repair", output=None)
    out = capsys.readouterr().out
    assert "RepairStateMachine" in out
    assert out.strip()


def test_png_requires_output():
    with pytest.raises(ValueError):
        render_state_machines(fmt="png", machine="repair", output=None)


def test_png_writes_file(tmp_path: Path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' binary not installed")
    out_path = tmp_path / "chart.png"
    render_state_machines(fmt="png", machine="repair", output=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_png_writes_both_files(tmp_path: Path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' binary not installed")
    base = tmp_path / "chart.png"
    render_state_machines(fmt="png", machine="both", output=base)
    first = base
    second = tmp_path / "chart_OptimizationStateMachine.png"
    assert first.exists()
    assert second.exists()
    assert first.stat().st_size > 0
    assert second.stat().st_size > 0
