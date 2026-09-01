"""Run persistence: atomic checkpoints, fail-closed resume.

Every completed step is checkpointed (write-temp + rename, so a crash mid-write
cannot corrupt the record). Resume replays completed steps from the record and
never re-executes (or re-bills) them.

Fail-closed rule, learned the expensive way in a production migration: resuming
against CHANGED inputs risks producing output attributed to the wrong task — so
resume verifies a fingerprint of (task, tool names) and REFUSES on mismatch.
A refused resume costs a re-run; a wrong resume costs correctness.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path


class ResumeRefused(Exception):
    pass


def fingerprint(task: str, tool_names: list[str]) -> str:
    blob = json.dumps({"task": task, "tools": sorted(tool_names)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class RunStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.state_path = self.dir / "state.json"

    def load(self, task: str, tool_names: list[str]) -> dict | None:
        """Prior state for resume, or None. Refuses a fingerprint mismatch."""
        if not self.state_path.exists():
            return None
        state = json.loads(self.state_path.read_text())
        fp = fingerprint(task, tool_names)
        if state.get("fingerprint") != fp:
            raise ResumeRefused(
                f"recorded run was for different inputs (fingerprint "
                f"{state.get('fingerprint')} != {fp}). Refusing to resume: "
                f"resuming against changed inputs risks wrong output. Start a "
                f"fresh run directory instead."
            )
        return state

    def save(self, state: dict) -> None:
        """Atomic full-state write: temp file + rename."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, default=_jsonable))
        os.replace(tmp, self.state_path)


def _jsonable(obj):
    try:
        return asdict(obj)
    except TypeError:
        return str(obj)
