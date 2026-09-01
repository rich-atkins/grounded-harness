"""Trajectory evals: record, replay, judge, gate — and the gate's failure modes."""
from __future__ import annotations

from grounded_harness import Agent, MockProvider, Tool, ToolCall, Turn
from grounded_harness.tools import Toolbox
from grounded_harness.trajectory import (
    GoldenResult, evaluate, gate, load_golden, replay, save_golden,
)


def _tool(answers: dict[str, str]) -> Tool:
    return Tool(name="lookup", description="",
                input_schema={"type": "object",
                              "properties": {"key": {"type": "string"}},
                              "required": ["key"]},
                fn=lambda key: answers.get(key, f"no entry for {key}"))


def _record(tmp_path, answers: dict[str, str]):
    script = [
        Turn(tool_calls=[ToolCall(id="c1", name="lookup", input={"key": "price"})],
             stop_reason="tool_use"),
        Turn(text="The price is £249."),
    ]
    run = Agent(MockProvider(script), Toolbox([_tool(answers)])).run("price?")
    path = tmp_path / "golden.json"
    save_golden(path, run, must_be_grounded=["£249"])
    return path


def test_record_replay_evaluate_green(tmp_path):
    path = _record(tmp_path, {"price": "£249"})
    golden = load_golden(path)
    run = replay(golden, Toolbox([_tool({"price": "£249"})]))
    res = evaluate(golden, run)
    assert all(res.families.values()), res.details


def test_changed_tool_data_breaks_groundedness(tmp_path):
    # The tool now answers differently: the recorded claim loses its source.
    path = _record(tmp_path, {"price": "£249"})
    golden = load_golden(path)
    run = replay(golden, Toolbox([_tool({"price": "£999"})]))
    res = evaluate(golden, run)
    assert res.families["groundedness"] is False
    assert any("not grounded" in d for d in res.details)


def test_erroring_tool_breaks_tool_correctness(tmp_path):
    path = _record(tmp_path, {"price": "£249"})
    golden = load_golden(path)

    def boom(key):
        raise RuntimeError("store offline")
    broken = Tool(name="lookup", description="", input_schema={}, fn=boom)
    res = evaluate(golden, replay(golden, Toolbox([broken])))
    assert res.families["tool_correctness"] is False


def _green(n: int) -> list[GoldenResult]:
    return [GoldenResult(name=f"g{i}", families={f: True for f in (
        "tool_correctness", "loop_efficiency", "groundedness", "budget_adherence")})
        for i in range(n)]


def test_gate_fails_closed_on_thin_coverage():
    report = gate(_green(2))  # below the floor of 3
    assert report.passed is False
    assert report.insufficient is not None
    assert report.rates == {}  # no confident numbers over thin evidence


def test_gate_absolute_bar_without_baseline():
    report = gate(_green(3))
    assert report.passed is True
    bad = _green(3)
    bad[0].families["groundedness"] = False
    assert gate(bad).passed is False


def test_gate_regression_vs_baseline():
    baseline = {"rates": {f: 1.0 for f in (
        "tool_correctness", "loop_efficiency", "groundedness", "budget_adherence")}}
    ok = gate(_green(3), baseline=baseline)
    assert ok.passed is True and ok.regressions == []
    bad = _green(3)
    bad[1].families["loop_efficiency"] = False
    report = gate(bad, baseline=baseline)
    assert report.passed is False
    assert any("loop_efficiency regressed" in r for r in report.regressions)
