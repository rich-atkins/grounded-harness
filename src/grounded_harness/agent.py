"""The agent loop: tool-calling, checkpointed, budgeted, and always accounted for.

A run ALWAYS ends in a Run object with an honest status — "completed",
"budget_exhausted", or "error" — carrying every step taken, the telemetry for
each, and the budget ledger. There is no code path that spends money without
leaving a record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .budget import Budget, BudgetExhausted
from .providers import ToolCall, Turn
from .store import RunStore
from .telemetry import Telemetry
from .tools import Toolbox, ToolResult


@dataclass
class Step:
    """One loop iteration: the assistant turn and any tool results it caused."""

    index: int
    turn: dict                    # serialised Turn
    tool_results: list[dict] = field(default_factory=list)


@dataclass
class Run:
    task: str
    status: str = "running"       # completed | budget_exhausted | error
    final_text: str = ""
    steps: list[Step] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    status_detail: str = ""
    resumed_steps: int = 0

    def as_dict(self) -> dict:
        return {
            "task": self.task, "status": self.status,
            "status_detail": self.status_detail, "final_text": self.final_text,
            "resumed_steps": self.resumed_steps, "budget": self.budget,
            "steps": [asdict(s) for s in self.steps],
        }


def _turn_to_dict(t: Turn) -> dict:
    return {
        "text": t.text, "stop_reason": t.stop_reason, "model": t.model,
        "input_tokens": t.input_tokens, "output_tokens": t.output_tokens,
        "latency_s": t.latency_s,
        "tool_calls": [{"id": c.id, "name": c.name, "input": c.input}
                       for c in t.tool_calls],
    }


def _turn_from_dict(d: dict) -> Turn:
    return Turn(
        text=d["text"], stop_reason=d["stop_reason"], model=d["model"],
        input_tokens=d["input_tokens"], output_tokens=d["output_tokens"],
        latency_s=d["latency_s"],
        tool_calls=[ToolCall(**c) for c in d["tool_calls"]],
    )


class Agent:
    def __init__(self, provider, toolbox: Toolbox | None = None,
                 budget: Budget | None = None, store: RunStore | None = None,
                 telemetry: Telemetry | None = None, system: str = ""):
        self.provider = provider
        self.toolbox = toolbox or Toolbox()
        self.budget = budget or Budget()
        self.store = store
        self.telemetry = telemetry or Telemetry()
        self.system = system

    # -- message assembly --------------------------------------------------------

    def _messages(self, task: str, steps: list[Step]) -> list[dict]:
        msgs: list[dict] = [{"role": "user", "content": self._task_text(task)}]
        for s in steps:
            turn = s.turn
            content: list[dict] = []
            if turn["text"]:
                content.append({"type": "text", "text": turn["text"]})
            for c in turn["tool_calls"]:
                content.append({"type": "tool_use", "id": c["id"],
                                "name": c["name"], "input": c["input"]})
            msgs.append({"role": "assistant", "content": content or turn["text"] or "…"})
            if s.tool_results:
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["tool_call_id"],
                     "content": r["content"],
                     **({"is_error": True} if r["is_error"] else {})}
                    for r in s.tool_results
                ]})
        return msgs

    def _task_text(self, task: str) -> str:
        return f"{self.system}\n\n{task}".strip() if self.system else task

    # -- the loop ----------------------------------------------------------------

    def run(self, task: str, run_id: str = "run") -> Run:
        run = Run(task=task)
        # Resume: replay recorded steps without re-executing anything.
        if self.store is not None:
            prior = self.store.load(task, self.toolbox.names())
            if prior is not None:
                run.steps = [Step(index=s["index"], turn=s["turn"],
                                  tool_results=s["tool_results"])
                             for s in prior["steps"]]
                run.resumed_steps = len(run.steps)
                self.budget.steps = len(run.steps)

        try:
            while True:
                self.budget.pre_step()  # step ceiling stops spend BEFORE it happens
                turn = self.provider.complete(
                    self._messages(task, run.steps), self.toolbox.specs())
                self.telemetry.record_turn(len(run.steps), "agent", turn)

                step = Step(index=len(run.steps), turn=_turn_to_dict(turn))
                if turn.tool_calls:
                    for call in turn.tool_calls:
                        result: ToolResult = self.toolbox.run(call)
                        step.tool_results.append({
                            "tool_call_id": result.tool_call_id,
                            "name": result.name, "content": result.content,
                            "is_error": result.is_error,
                        })
                run.steps.append(step)
                self._checkpoint(task, run)
                # Cost/token ceilings charge AFTER the step is recorded: a paid
                # turn always leaves a record, then the run ends at the boundary.
                from .providers import turn_cost_usd
                self.budget.charge(turn_cost_usd(turn), turn.output_tokens)

                if not turn.tool_calls:
                    run.status = "completed"
                    run.final_text = turn.text
                    break
        except BudgetExhausted as e:
            run.status = "budget_exhausted"
            run.status_detail = e.reason
            run.final_text = self._partial_summary(run)
        except Exception as e:  # harness/provider error: recorded, not swallowed
            run.status = "error"
            run.status_detail = f"{type(e).__name__}: {e}"

        run.budget = self.budget.snapshot()
        self._checkpoint(task, run)
        return run

    def _partial_summary(self, run: Run) -> str:
        tools_used = [c["name"] for s in run.steps for c in s.turn["tool_calls"]]
        return (f"[budget exhausted after {len(run.steps)} steps; "
                f"tools used: {', '.join(tools_used) or 'none'}; partial "
                f"trajectory preserved]")

    def _checkpoint(self, task: str, run: Run) -> None:
        if self.store is None:
            return
        from .store import fingerprint
        self.store.save({
            "fingerprint": fingerprint(task, self.toolbox.names()),
            "task": task, "status": run.status,
            "steps": [asdict(s) for s in run.steps],
            "budget": self.budget.snapshot(),
        })
