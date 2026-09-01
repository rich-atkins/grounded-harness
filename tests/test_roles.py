"""Planner → workers → critic, fully deterministic on the mock provider."""
from __future__ import annotations

from grounded_harness import Agent, Budget, MockProvider, ToolCall, Turn
from grounded_harness.roles import Team
from grounded_harness.tools import Toolbox


def _planner(subtasks: list[str]) -> Agent:
    script = [
        Turn(tool_calls=[ToolCall(id="p1", name="submit_plan",
                                  input={"subtasks": subtasks})],
             stop_reason="tool_use"),
        Turn(text="plan submitted"),
    ]
    return Agent(MockProvider(script), Toolbox())


def _worker_factory(answers: dict[int, str]):
    def make(i: int, subtask: str) -> Agent:
        return Agent(MockProvider([Turn(text=answers[i])]), Toolbox())
    return make


def _critic(verdict: str = "approve", notes: str = "coherent") -> Agent:
    script = [
        Turn(tool_calls=[ToolCall(id="r1", name="submit_review",
                                  input={"verdict": verdict, "notes": notes})],
             stop_reason="tool_use"),
        Turn(text="review submitted"),
    ]
    return Agent(MockProvider(script), Toolbox())


def test_team_completes_end_to_end():
    team = Team(_planner(["find the price", "find the warranty"]),
                _worker_factory({0: "price is £249", 1: "warranty is five years"}),
                _critic())
    result = team.run("summarise Widget Pro's terms")
    assert result.status == "completed"
    assert result.subtasks == ["find the price", "find the warranty"]
    assert [w["final_text"] for w in result.worker_runs] == \
        ["price is £249", "warranty is five years"]
    assert result.verdict == "approve"


def test_unsubmitted_plan_fails_loudly():
    # Planner talks ABOUT a plan but never calls the tool: not a plan.
    planner = Agent(MockProvider([Turn(text="I would split this into two parts.")]),
                    Toolbox())
    team = Team(planner, _worker_factory({}), _critic())
    result = team.run("task")
    assert result.status == "plan_failed"
    assert result.worker_runs == []  # no work happens on a failed hand-off


def test_unsubmitted_review_fails_loudly():
    critic = Agent(MockProvider([Turn(text="Looks fine to me.")]), Toolbox())
    team = Team(_planner(["one thing"]), _worker_factory({0: "done"}), critic)
    result = team.run("task")
    assert result.status == "review_failed"


def test_budget_exhausted_worker_is_not_a_team_failure():
    def make(i, subtask):
        endless = [Turn(tool_calls=[ToolCall(id=f"c{n}", name="nope", input={})],
                        stop_reason="tool_use") for n in range(5)]
        return Agent(MockProvider(endless), Toolbox(), budget=Budget(max_steps=2))
    team = Team(_planner(["hard thing"]), make, _critic("revise", "incomplete"))
    result = team.run("task")
    # Partial work is real work, honestly labelled; the critic judges it.
    assert result.status == "completed"
    assert result.worker_runs[0]["status"] == "budget_exhausted"
    assert result.verdict == "revise"
