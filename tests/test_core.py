"""Core loop, budget, and resume behaviour. All offline on the mock provider."""
from __future__ import annotations

import pytest

from grounded_harness import (
    Agent, Budget, BudgetExhausted, MockProvider, ResumeRefused, RunStore,
    Tool, ToolCall, Turn,
)
from grounded_harness.tools import Toolbox


def _lookup_tool(answers: dict[str, str]) -> Tool:
    return Tool(
        name="lookup", description="look a key up",
        input_schema={"type": "object", "properties": {"key": {"type": "string"}},
                      "required": ["key"]},
        fn=lambda key: answers.get(key, f"no entry for {key}"),
    )


def _script_two_step() -> list[Turn]:
    return [
        Turn(tool_calls=[ToolCall(id="c1", name="lookup", input={"key": "price"})],
             stop_reason="tool_use", input_tokens=100, output_tokens=20),
        Turn(text="The price is £249.", input_tokens=150, output_tokens=15),
    ]


def test_loop_runs_tools_and_completes():
    agent = Agent(MockProvider(_script_two_step()),
                  Toolbox([_lookup_tool({"price": "£249"})]))
    run = agent.run("what is the price?")
    assert run.status == "completed"
    assert run.final_text == "The price is £249."
    assert run.steps[0].tool_results[0]["content"] == "£249"
    assert run.budget["cost_known"] is False  # mock model has no price: unknown != zero


def test_tool_failure_is_data_not_crash():
    def boom(key):
        raise ValueError("db down")
    tool = Tool(name="lookup", description="", input_schema={}, fn=boom)
    agent = Agent(MockProvider(_script_two_step()), Toolbox([tool]))
    run = agent.run("q")
    assert run.status == "completed"
    assert run.steps[0].tool_results[0]["is_error"] is True
    assert "db down" in run.steps[0].tool_results[0]["content"]


def test_budget_step_ceiling_yields_honest_partial():
    script = [Turn(tool_calls=[ToolCall(id=f"c{i}", name="lookup", input={"key": "x"})],
                   stop_reason="tool_use") for i in range(5)]
    agent = Agent(MockProvider(script), Toolbox([_lookup_tool({})]),
                  budget=Budget(max_steps=3))
    run = agent.run("loop forever")
    assert run.status == "budget_exhausted"
    assert "step ceiling" in run.status_detail
    assert len(run.steps) <= 3
    assert "partial trajectory preserved" in run.final_text


def test_cost_ceiling_fails_closed_on_unknown_usage():
    # Turns with unknown tokens + a cost ceiling: the harness must refuse to
    # continue rather than treat unknown as free. Unknown is not zero.
    script = [Turn(tool_calls=[ToolCall(id="c1", name="lookup", input={"key": "x"})],
                   stop_reason="tool_use")]  # no token counts
    agent = Agent(MockProvider(script), Toolbox([_lookup_tool({})]),
                  budget=Budget(max_cost_usd=1.0))
    run = agent.run("q")
    assert run.status == "budget_exhausted"
    assert "unverifiable budget" in run.status_detail


def test_resume_replays_without_reexecuting(tmp_path):
    calls = []
    def counting(key):
        calls.append(key)
        return "£249"
    tool = Tool(name="lookup", description="", input_schema={}, fn=counting)
    store = RunStore(tmp_path / "run1")

    # First run: budget stops it after the tool step.
    a1 = Agent(MockProvider(_script_two_step()), Toolbox([tool]),
               budget=Budget(max_steps=1), store=store)
    r1 = a1.run("what is the price?")
    assert r1.status == "budget_exhausted"
    assert calls == ["price"]

    # Resume with room: the recorded step replays, tool NOT re-executed.
    a2 = Agent(MockProvider([_script_two_step()[1]]), Toolbox([tool]),
               budget=Budget(max_steps=5), store=store)
    r2 = a2.run("what is the price?")
    assert r2.status == "completed"
    assert r2.resumed_steps == 1
    assert calls == ["price"]  # still exactly one execution


def test_resume_refuses_changed_inputs(tmp_path):
    store = RunStore(tmp_path / "run1")
    tool = _lookup_tool({"price": "£249"})
    a1 = Agent(MockProvider(_script_two_step()), Toolbox([tool]), store=store)
    a1.run("what is the price?")
    a2 = Agent(MockProvider(_script_two_step()), Toolbox([tool]), store=store)
    with pytest.raises(ResumeRefused):
        a2.store.load("a DIFFERENT task", a2.toolbox.names())
