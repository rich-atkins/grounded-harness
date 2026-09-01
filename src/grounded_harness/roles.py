"""Multi-agent primitives: planner → workers (parallel) → critic.

Deliberately minimal — three roles, explicit hand-offs, no framework cosplay.
Each role IS an Agent (its own provider, tools, budget, store), so everything
the single-agent loop guarantees (telemetry, budgets, honest statuses) holds
per role with no special cases.

Structured hand-offs use tools, not text parsing: the planner must CALL
``submit_plan``; the critic must CALL ``submit_review``. A role that never
calls its tool produces a failed hand-off, loudly — a plan that has to be
scraped out of prose with a regex is a plan you cannot gate on.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .agent import Agent, Run
from .tools import Tool, Toolbox


PLAN_SCHEMA = {
    "type": "object",
    "properties": {"subtasks": {"type": "array", "items": {"type": "string"},
                                "minItems": 1}},
    "required": ["subtasks"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "revise"]},
        "notes": {"type": "string"},
    },
    "required": ["verdict", "notes"],
}


@dataclass
class TeamRun:
    task: str
    status: str = "running"    # completed | plan_failed | review_failed | worker_failed
    subtasks: list[str] = field(default_factory=list)
    planner_run: dict = field(default_factory=dict)
    worker_runs: list[dict] = field(default_factory=list)
    critic_run: dict = field(default_factory=dict)
    verdict: str = ""
    verdict_notes: str = ""

    def as_dict(self) -> dict:
        return {
            "task": self.task, "status": self.status, "subtasks": self.subtasks,
            "verdict": self.verdict, "verdict_notes": self.verdict_notes,
            "planner_run": self.planner_run, "worker_runs": self.worker_runs,
            "critic_run": self.critic_run,
        }


class Team:
    """planner/worker_factory/critic are caller-built Agents so every role's
    provider, tools and budget are explicit. ``worker_factory(i, subtask)``
    returns a fresh Agent per subtask (workers run in parallel threads)."""

    def __init__(self, planner: Agent,
                 worker_factory: Callable[[int, str], Agent],
                 critic: Agent, max_parallel: int = 4):
        self.planner = planner
        self.worker_factory = worker_factory
        self.critic = critic
        self.max_parallel = max_parallel

    def run(self, task: str) -> TeamRun:
        team = TeamRun(task=task)

        # -- plan ---------------------------------------------------------------
        captured: dict = {}

        def submit_plan(subtasks: list) -> str:
            captured["subtasks"] = [str(s) for s in subtasks]
            return f"plan recorded ({len(subtasks)} subtasks)"

        self.planner.toolbox.add(Tool(
            name="submit_plan",
            description="Submit the final plan as a list of self-contained subtasks.",
            input_schema=PLAN_SCHEMA, fn=submit_plan))
        planner_run = self.planner.run(
            f"Plan this task as a short list of self-contained subtasks, then "
            f"submit it with the submit_plan tool.\n\nTASK: {task}")
        team.planner_run = planner_run.as_dict()
        if "subtasks" not in captured:
            team.status = "plan_failed"
            return team  # a plan that was never submitted is not a plan
        team.subtasks = captured["subtasks"]

        # -- work (parallel) ----------------------------------------------------
        agents = [self.worker_factory(i, s) for i, s in enumerate(team.subtasks)]
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            runs: list[Run] = list(pool.map(
                lambda pair: pair[0].run(pair[1]), zip(agents, team.subtasks)))
        team.worker_runs = [r.as_dict() for r in runs]
        if any(r.status == "error" for r in runs):
            team.status = "worker_failed"
            return team
        # budget_exhausted workers are NOT a team failure: their partial output
        # is real work, honestly labelled; the critic judges what exists.

        # -- review -------------------------------------------------------------
        review: dict = {}

        def submit_review(verdict: str, notes: str) -> str:
            review.update(verdict=verdict, notes=notes)
            return "review recorded"

        self.critic.toolbox.add(Tool(
            name="submit_review",
            description="Submit the final review verdict for the assembled work.",
            input_schema=REVIEW_SCHEMA, fn=submit_review))
        assembled = "\n\n".join(
            f"SUBTASK {i}: {s}\nSTATUS: {r.status}\nRESULT:\n{r.final_text}"
            for i, (s, r) in enumerate(zip(team.subtasks, runs)))
        critic_run = self.critic.run(
            f"Review the assembled work against the original task, then submit "
            f"your verdict with the submit_review tool.\n\nTASK: {task}\n\n{assembled}")
        team.critic_run = critic_run.as_dict()
        if "verdict" not in review:
            team.status = "review_failed"
            return team

        team.verdict = review["verdict"]
        team.verdict_notes = review["notes"]
        team.status = "completed"
        return team
