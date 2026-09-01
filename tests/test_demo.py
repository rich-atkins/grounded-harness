"""The demo IS the sabotage suite: green for the right reasons, red for the right reasons."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVER = Path(sys.executable).parent / "grounded-mcp"
pytestmark = pytest.mark.skipif(not SERVER.exists(),
                                reason="grounded-mcp not installed (dev extra)")


def test_demo_gate_passes():
    from grounded_harness.demo import run_demo
    assert run_demo() == 0


def test_sabotage_broken_tool_fails_gate(capsys):
    from grounded_harness.demo import run_demo
    assert run_demo(sabotage="broken-tool") == 1
    out = capsys.readouterr().out
    assert "not grounded" in out
    assert "GATE: FAIL" in out


def test_sabotage_thin_evidence_refuses(capsys):
    from grounded_harness.demo import run_demo
    assert run_demo(sabotage="thin-evidence") == 1
    out = capsys.readouterr().out
    assert "insufficient evidence" in out
