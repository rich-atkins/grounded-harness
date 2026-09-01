"""Trajectory evals: judge the WHOLE run, gate on regression, fail closed on thin evidence.

A golden trajectory is a recorded run plus expectations, kept under
``evals/golden/approved/``. Replaying it reconstructs the mock script from the
recorded turns and re-executes the run — the model side is deterministic, the
TOOLS run for real. That makes replay a regression harness for everything except
the model: harness code, tool code, tool data.

Four eval families per golden:
  tool_correctness   the tool-call sequence (names + inputs) matches the record,
                     and no tool result came back as an error
  loop_efficiency    the run took no more steps than the record allows
  groundedness       every claim listed in ``must_be_grounded`` appears BOTH in
                     the final text AND in some tool result (a claim with no
                     tool-derived source is not grounded, wherever it came from)
  budget_adherence   the run ended with the status the golden expects

The gate aggregates across goldens vs a committed baseline and FAILS CLOSED when
coverage is below the evidence floor: a gate that judged too little to mean
anything must say so, not PASS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent, Run, _turn_from_dict
from .budget import Budget
from .providers import MockProvider
from .tools import Toolbox

MIN_EVIDENCE_GOLDENS = 3


@dataclass
class GoldenResult:
    name: str
    families: dict[str, bool] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)


def load_golden(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_golden(path: Path, run: Run, *, must_be_grounded: list[str] | None = None,
                expected_status: str = "completed", max_steps_tolerance: int = 0) -> None:
    """Record a run as a golden candidate (promotion to approved/ is a human act)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "task": run.task,
        "expected_status": expected_status,
        "must_be_grounded": must_be_grounded or [],
        "max_steps_tolerance": max_steps_tolerance,
        "run": run.as_dict(),
    }, indent=2))


def replay(golden: dict, toolbox: Toolbox, budget: Budget | None = None) -> Run:
    """Re-execute the golden's run: scripted model turns, REAL tool execution."""
    script = [_turn_from_dict(s["turn"]) for s in golden["run"]["steps"]]
    agent = Agent(MockProvider(script), toolbox, budget=budget or Budget())
    return agent.run(golden["task"])


def evaluate(golden: dict, run: Run) -> GoldenResult:
    name = golden.get("name") or golden["task"][:40]
    res = GoldenResult(name=name)
    ref_steps = golden["run"]["steps"]

    # -- tool_correctness --------------------------------------------------------
    ref_calls = [(c["name"], json.dumps(c["input"], sort_keys=True))
                 for s in ref_steps for c in s["turn"]["tool_calls"]]
    got_calls = [(c["name"], json.dumps(c["input"], sort_keys=True))
                 for s in run.steps for c in s.turn["tool_calls"]]
    errors = [r["name"] for s in run.steps for r in s.tool_results if r["is_error"]]
    ok = ref_calls == got_calls and not errors
    res.families["tool_correctness"] = ok
    if not ok:
        if ref_calls != got_calls:
            res.details.append(f"tool sequence diverged: expected {ref_calls}, got {got_calls}")
        for e in errors:
            res.details.append(f"tool errored during replay: {e}")

    # -- loop_efficiency ---------------------------------------------------------
    allowed = len(ref_steps) + int(golden.get("max_steps_tolerance", 0))
    ok = len(run.steps) <= allowed
    res.families["loop_efficiency"] = ok
    if not ok:
        res.details.append(f"took {len(run.steps)} steps, allowed {allowed}")

    # -- groundedness ------------------------------------------------------------
    tool_blob = "\n".join(r["content"] for s in run.steps for r in s.tool_results)
    misses = [claim for claim in golden.get("must_be_grounded", [])
              if claim not in run.final_text or claim not in tool_blob]
    res.families["groundedness"] = not misses
    for m in misses:
        where = []
        if m not in run.final_text:
            where.append("final text")
        if m not in tool_blob:
            where.append("any tool result")
        res.details.append(f"claim not grounded: {m!r} missing from {' and '.join(where)}")

    # -- budget_adherence --------------------------------------------------------
    expected = golden.get("expected_status", "completed")
    ok = run.status == expected
    res.families["budget_adherence"] = ok
    if not ok:
        res.details.append(f"status {run.status!r}, expected {expected!r}")
    return res


FAMILIES = ("tool_correctness", "loop_efficiency", "groundedness", "budget_adherence")


@dataclass
class GateReport:
    n: int
    rates: dict[str, float] = field(default_factory=dict)
    insufficient: str | None = None
    failures: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    passed: bool = False


def gate(results: list[GoldenResult], baseline: dict | None = None,
         min_goldens: int = MIN_EVIDENCE_GOLDENS) -> GateReport:
    """Aggregate, compare to baseline, fail closed on thin coverage."""
    report = GateReport(n=len(results))
    if len(results) < min_goldens:
        report.insufficient = (f"n={len(results)} goldens, need >={min_goldens}: "
                               f"refusing to PASS on insufficient evidence")
        report.passed = False
        return report
    for fam in FAMILIES:
        passes = sum(1 for r in results if r.families.get(fam))
        report.rates[fam] = round(passes / len(results), 4)
    for r in results:
        for d in r.details:
            report.failures.append(f"[{r.name}] {d}")
    if baseline:
        for fam, base_rate in baseline.get("rates", {}).items():
            got = report.rates.get(fam, 0.0)
            if got < base_rate - 1e-9:
                report.regressions.append(f"{fam} regressed: {base_rate} -> {got}")
    if baseline is None:
        # No baseline committed yet: the bar is absolute — every family perfect.
        report.passed = all(v == 1.0 for v in report.rates.values())
    else:
        report.passed = not report.regressions
    return report
