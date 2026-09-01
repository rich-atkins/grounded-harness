"""CLI: the demo (with its sabotage modes) and the golden-gate runner."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .budget import Budget
from .trajectory import MIN_EVIDENCE_GOLDENS, evaluate, gate, load_golden, replay


def cmd_demo(args) -> int:
    from .demo import run_demo
    return run_demo(sabotage=args.sabotage)


def _toolbox_from(spec: str):
    """Import 'package.module:factory' and call it — tools are code, so the
    eval runner is told where the code lives rather than pretending JSON
    could describe it."""
    mod_name, _, fn_name = spec.partition(":")
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return fn()


def cmd_evals(args) -> int:
    toolbox = _toolbox_from(args.toolbox)
    goldens = sorted(Path(args.golden_dir).glob("*.json"))
    results = []
    for path in goldens:
        golden = load_golden(path)
        run = replay(golden, toolbox, budget=Budget(max_steps=args.max_steps))
        results.append(evaluate(golden, run))

    baseline = None
    if args.baseline and Path(args.baseline).exists() and not args.write_baseline:
        baseline = json.loads(Path(args.baseline).read_text())
    report = gate(results, baseline=baseline, min_goldens=args.min_goldens)

    print(f"grounded-harness evals  ({len(results)} goldens)")
    if report.insufficient:
        print(f"  GATE: FAIL — {report.insufficient}")
        return 1
    for fam, rate in report.rates.items():
        print(f"  {fam:<18} {rate:.2f}")
    for line in report.failures + report.regressions:
        print(f"  ! {line}")
    if args.write_baseline:
        Path(args.baseline).write_text(json.dumps(
            {"rates": report.rates}, indent=2) + "\n")
        print(f"  baseline written -> {args.baseline}")
    print(f"  GATE: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="grounded-harness",
        description="An agent harness where evaluation is a runtime property.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="offline demo over grounded-mcp's ACME vault")
    d.add_argument("--sabotage", choices=["broken-tool", "thin-evidence"],
                   help="watch the gate fail for the right reasons")
    d.set_defaults(fn=cmd_demo)

    e = sub.add_parser("evals", help="replay goldens, judge, gate")
    e.add_argument("--toolbox", required=True,
                   help="import spec 'package.module:factory' returning a Toolbox")
    e.add_argument("--golden-dir", required=True)
    e.add_argument("--baseline", default=None)
    e.add_argument("--write-baseline", action="store_true")
    e.add_argument("--min-goldens", type=int, default=MIN_EVIDENCE_GOLDENS)
    e.add_argument("--max-steps", type=int, default=20)
    e.set_defaults(fn=cmd_evals)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
