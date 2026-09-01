"""Per-call telemetry: one JSONL line per model turn, one choke point.

The JSONL is the portable record; OpenTelemetry export is an optional view over
the same events (lazy import, env-gated via GH_OTEL=1), never a replacement.
Missing usage is recorded as null — unknown is not zero, and downstream
aggregation must treat null as "insufficient evidence", not as free.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .providers import Turn, turn_cost_usd


class Telemetry:
    def __init__(self, path: Path | None = None, run_id: str = "run"):
        self.path = path
        self.run_id = run_id
        self._otel = os.environ.get("GH_OTEL", "") in ("1", "true", "yes")

    def record_turn(self, step: int, role: str, turn: Turn) -> dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "role": role,
            "model": turn.model,
            "input_tokens": turn.input_tokens,     # None = unknown, kept as null
            "output_tokens": turn.output_tokens,
            "latency_s": round(turn.latency_s, 3) if turn.latency_s is not None else None,
            "cost_usd": turn_cost_usd(turn),
            "tool_calls": [c.name for c in turn.tool_calls],
            "stop_reason": turn.stop_reason,
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        if self._otel:
            self._emit_otel(entry)
        return entry

    def _emit_otel(self, entry: dict) -> None:
        try:
            from opentelemetry import trace
        except ImportError:
            return  # optional view; absence is not an error
        tracer = trace.get_tracer("grounded-harness")
        with tracer.start_as_current_span(f"turn.{entry['role']}") as span:
            for k in ("run_id", "step", "model", "stop_reason"):
                span.set_attribute(f"gh.{k}", str(entry[k]))
            for k in ("input_tokens", "output_tokens", "cost_usd", "latency_s"):
                if entry[k] is not None:
                    span.set_attribute(f"gh.{k}", entry[k])
