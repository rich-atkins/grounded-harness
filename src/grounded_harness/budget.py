"""Budgets: ceilings the run knows about, not surprises it dies from.

A budget-exhausted run is a VALID OUTCOME with partial results and an honest
status — not an exception splattered over a half-finished trajectory. The
exception exists for the loop's control flow only; callers receive a completed
``Run`` whose status says what happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExhausted(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Budget:
    max_steps: int = 20
    max_cost_usd: float | None = None
    max_output_tokens: int | None = None

    # -- accounting (filled during the run) -------------------------------------
    steps: int = 0
    cost_usd: float = 0.0
    output_tokens: int = 0
    cost_known: bool = field(default=True)
    """False once any turn had unknown token counts: the cost figure is then a
    LOWER BOUND, and a lower bound must never be allowed to pass a ceiling check
    as if it were a total. Unknown is not zero."""

    def pre_step(self) -> None:
        """Called BEFORE a provider call: the step ceiling must stop spend
        before it happens, not discard a paid turn after."""
        if self.steps >= self.max_steps:
            raise BudgetExhausted(f"step ceiling reached ({self.max_steps})")

    def charge(self, cost: float | None, out_tokens: int | None) -> None:
        """Called AFTER the step is recorded: paid work is always preserved;
        cost/token ceilings end the run at the next boundary."""
        self.steps += 1
        if cost is None or out_tokens is None:
            self.cost_known = False
        else:
            self.cost_usd += cost
            self.output_tokens += out_tokens
        self._check()

    def _check(self) -> None:
        if self.max_cost_usd is not None:
            if not self.cost_known:
                # Fail closed: with unknown costs we cannot prove we are under
                # the ceiling, and a cost ceiling that cannot fire is decoration.
                raise BudgetExhausted(
                    "cost ceiling set but token usage is unknown for at least one "
                    "turn: refusing to continue on an unverifiable budget"
                )
            if self.cost_usd >= self.max_cost_usd:
                raise BudgetExhausted(f"cost ceiling reached (${self.max_cost_usd})")
        if self.max_output_tokens is not None and self.output_tokens >= self.max_output_tokens:
            raise BudgetExhausted(f"output-token ceiling reached ({self.max_output_tokens})")

    def snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "cost_usd": round(self.cost_usd, 6),
            "cost_known": self.cost_known,
            "output_tokens": self.output_tokens,
            "max_steps": self.max_steps,
            "max_cost_usd": self.max_cost_usd,
            "max_output_tokens": self.max_output_tokens,
        }
