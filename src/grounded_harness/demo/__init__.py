"""The offline money demo: an agent over grounded-mcp's ACME vault, fully evaluated.

Model turns are scripted (deterministic), tools are REAL — a live grounded-mcp
server over stdio answering from its committed demo vault. No network, no keys.

Three modes:
    normal                    replay the three goldens, gate vs baseline -> PASS
    --sabotage broken-tool    read_note returns garbage -> groundedness/tool-
                              correctness go red -> gate FAILS (exit 1)
    --sabotage thin-evidence  only one golden replayed -> the gate refuses to
                              PASS on insufficient evidence (exit 1)

Both failures are the point: a green scorecard only means something if you have
watched the gate go red for the right reasons.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from ..agent import Agent
from ..budget import Budget
from ..providers import MockProvider, ToolCall, Turn
from ..tools import MCPServerSpec, MCPToolbox, Tool, Toolbox, mcp_toolbox
from ..trajectory import evaluate, gate, load_golden, replay, save_golden

DEMO_DIR = Path(__file__).parent
GOLDEN_DIR = DEMO_DIR / "goldens"
BASELINE = DEMO_DIR / "baseline.json"


def acme_toolbox() -> Toolbox:
    """Toolbox proxying to a real grounded-mcp server over the ACME demo vault.

    The vault is VENDORED inside this package (demo/vault): the demo must work
    from any install (pip-from-git puts no repo checkout on disk), and pinning
    our own copy keeps the goldens stable against upstream vault edits.
    """
    server = shutil.which("grounded-mcp") or str(
        Path(sys.executable).parent / "grounded-mcp")
    import os
    env = dict(os.environ)
    env["GROUNDED_VAULT"] = str(DEMO_DIR / "vault")
    env["GROUNDED_PROFILE"] = "staff"
    return mcp_toolbox(MCPServerSpec(command=server, env=env))


class _SabotagedToolbox(Toolbox):
    """Wraps a toolbox; read_note returns garbage. The 'broken-tool' sabotage."""

    def __init__(self, inner: Toolbox):
        super().__init__()
        self._inner = inner

    def specs(self):
        return self._inner.specs()

    def names(self):
        return self._inner.names()

    def run(self, call):
        result = self._inner.run(call)
        if call.name == "read_note":
            result.content = "{}"  # the note evaporates; claims lose their source
        return result


# The three demo tasks: (name, task, script, must_be_grounded)
def _scripts() -> list[tuple[str, str, list[Turn], list[str]]]:
    def s(cid, query, citation, answer):
        return [
            Turn(tool_calls=[ToolCall(id=f"{cid}-s", name="search",
                                      input={"query": query, "k": 2})],
                 stop_reason="tool_use"),
            Turn(tool_calls=[ToolCall(id=f"{cid}-r", name="read_note",
                                      input={"citation": citation,
                                             "section_only": False})],
                 stop_reason="tool_use"),
            Turn(text=answer),
        ]
    return [
        ("widget-price", "What does Widget Pro cost?",
         s("g1", "widget pro pricing", "public/products/widget-pro.md",
           "Widget Pro lists at £249 per unit."), ["£249"]),
        ("rollback-time", "How long does a rollback take?",
         s("g2", "rollback deploy", "internal/engineering/deploy-process.md",
           "Rollback is one click and takes about 90 seconds."), ["90 seconds"]),
        ("gadget-colours", "What colours does Gadget Lite come in?",
         s("g3", "gadget lite colours", "public/products/gadget-lite.md",
           "Gadget Lite comes in slate, moss, and clay."), ["slate", "moss", "clay"]),
    ]


def generate_goldens() -> None:
    """Record the three demo runs as goldens (dev-time; committed to the repo)."""
    toolbox = acme_toolbox()
    for name, task, script, grounded in _scripts():
        run = Agent(MockProvider(script), toolbox, budget=Budget(max_steps=6)).run(task)
        assert run.status == "completed", (name, run.status, run.status_detail)
        save_golden(GOLDEN_DIR / f"{name}.json", run, must_be_grounded=grounded)
        print(f"recorded golden: {name}")


def run_demo(sabotage: str | None = None) -> int:
    goldens = sorted(GOLDEN_DIR.glob("*.json"))
    if sabotage == "thin-evidence":
        goldens = goldens[:1]
    toolbox = acme_toolbox()
    if sabotage == "broken-tool":
        toolbox = _SabotagedToolbox(toolbox)

    results = []
    for path in goldens:
        golden = load_golden(path)
        run = replay(golden, toolbox, budget=Budget(max_steps=6))
        res = evaluate(golden, run)
        results.append(res)
        marks = "  ".join(f"{k}={'ok' if v else 'FAIL'}"
                          for k, v in res.families.items())
        print(f"  {path.stem:<16} {marks}")

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    report = gate(results, baseline=baseline)
    print()
    if report.insufficient:
        print(f"  GATE: FAIL — {report.insufficient}")
        return 1
    for fam, rate in report.rates.items():
        print(f"  {fam:<18} {rate:.2f}")
    for line in report.failures + report.regressions:
        print(f"  ! {line}")
    print(f"\n  GATE: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1
