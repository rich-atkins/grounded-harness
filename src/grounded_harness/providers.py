"""Model providers behind one seam.

Two providers ship: a deterministic ``MockProvider`` (scripted turns — the offline
demo, the test suite, and golden-trajectory replay all run on it) and a lazy-imported
``AnthropicProvider``. The seam is deliberate and small so other backends can be
added without touching the loop; it also means the whole harness is testable with
zero network and zero credentials, which is what makes the eval gates honest.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    """One assistant turn: text and/or tool calls, plus usage telemetry."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int | None = None   # None means unknown, and unknown is not zero
    output_tokens: int | None = None
    latency_s: float | None = None
    model: str = "mock"


class MockProvider:
    """Deterministic provider: consumes a scripted list of turns in order.

    Determinism is a feature, not a shortcut: golden-trajectory replay and the CI
    gates only mean something if the same script always produces the same run.
    """

    def __init__(self, script: list[Turn]):
        self._script = list(script)
        self._cursor = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> Turn:
        if self._cursor >= len(self._script):
            raise RuntimeError(
                f"mock script exhausted after {self._cursor} turns; the agent asked "
                f"for another turn. The script and the loop disagree — fix the script, "
                f"do not pad it."
            )
        turn = self._script[self._cursor]
        self._cursor += 1
        return turn


# Default tiers. Overridable by env; the router in future versions picks per task.
QUALITY_MODEL = os.environ.get("GH_QUALITY_MODEL", "claude-opus-5")
CHEAP_MODEL = os.environ.get("GH_CHEAP_MODEL", "claude-haiku-4-5")


class AnthropicProvider:
    """Live provider on the Anthropic SDK. Lazy import: mock-only use needs no dep."""

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        self.model = model or QUALITY_MODEL
        self.max_tokens = max_tokens
        self._client = None

    def _anthropic(self):
        if self._client is None:
            from anthropic import Anthropic  # reads ANTHROPIC_API_KEY
            self._client = Anthropic()
        return self._client

    def complete(self, messages: list[dict], tools: list[dict]) -> Turn:
        t0 = time.monotonic()
        msg = self._anthropic().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            **({"tools": tools} if tools else {}),
        )
        latency = time.monotonic() - t0
        text_parts, calls = [], []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        return Turn(
            text="".join(text_parts),
            tool_calls=calls,
            stop_reason=msg.stop_reason or "end_turn",
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_s=latency,
            model=self.model,
        )


# List prices per million tokens (input, output) for cost telemetry.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def turn_cost_usd(turn: Turn) -> float | None:
    """Cost of a turn, or None when tokens are unknown. Unknown is not zero."""
    if turn.input_tokens is None or turn.output_tokens is None:
        return None
    prices = PRICES_PER_MTOK.get(turn.model)
    if prices is None:
        return None
    return (turn.input_tokens * prices[0] + turn.output_tokens * prices[1]) / 1_000_000
