"""grounded-harness — an agent harness where evaluation is a runtime property.

Three governing rules, enforced in code rather than documentation:
  * unknown is not zero — missing telemetry reports as missing
  * evidence floors — an eval aggregate over too few runs says "insufficient"
  * every gate can fail — and ships with the sabotage test that proves it
"""

__version__ = "0.1.0"

from .agent import Agent, Run, Step
from .budget import Budget, BudgetExhausted
from .providers import AnthropicProvider, MockProvider, Turn, ToolCall
from .store import RunStore, ResumeRefused
from .tools import Tool, ToolError

__all__ = [
    "Agent", "Run", "Step", "Budget", "BudgetExhausted",
    "AnthropicProvider", "MockProvider", "Turn", "ToolCall",
    "RunStore", "ResumeRefused", "Tool", "ToolError",
]
