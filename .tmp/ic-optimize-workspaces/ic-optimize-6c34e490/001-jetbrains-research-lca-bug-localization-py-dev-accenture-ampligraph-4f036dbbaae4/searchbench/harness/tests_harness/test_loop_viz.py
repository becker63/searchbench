from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.tools.loop_viz import render_state_machines  # noqa: E402


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


def test_png_both_writes_repair_file_only(tmp_path: Path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' binary not installed")
    base = tmp_path / "chart.png"
    render_state_machines(fmt="png", machine="both", output=base)
    assert base.exists()
    assert base.stat().st_size > 0
    assert not (tmp_path / "chart_OptimizationStateMachine.png").exists()


def test_optimization_machine_render_is_removed():
    with pytest.raises(ValueError, match="explicit optimize loop"):
        render_state_machines(fmt="mermaid", machine="optimization", output=None)
