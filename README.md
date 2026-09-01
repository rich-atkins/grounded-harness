# grounded-harness

**An agent harness where evaluation is a runtime property: every run evaluated, every run resumable, every gate able to fail.**

Agent harnesses are everywhere. Eval harnesses exist separately. This project's bet is that they belong in the same spine: a run should not be able to complete without knowing how well it went, should not lose money to a crash, and should not silently drift from its recorded baseline.

Three governing rules, enforced in code rather than documentation:

1. **Unknown is not zero.** Missing token counts are recorded as missing, and a cost ceiling that cannot be verified refuses to pass itself ("unverifiable budget" ends the run rather than treating unknown spend as free).
2. **Evidence floors.** An eval aggregate over too few runs reports `insufficient`, never a confident number.
3. **Every gate can fail — and ships with the sabotage test that proves it.**

## Status: v0.1 in build

| Surface | State |
|---|---|
| Tool-calling loop (mock + Anthropic providers) | ✅ built, tested |
| MCP bridge (stdio) — agents consume MCP servers as tools | ✅ built, integration-tested against [grounded-mcp](https://github.com/rich-atkins/grounded-mcp) |
| Budgets (step / cost / token ceilings, honest partial results) | ✅ built, tested |
| Checkpoint / resume (atomic, fail-closed on changed inputs) | ✅ built, tested |
| Per-turn telemetry JSONL (+ optional OpenTelemetry) | ✅ built |
| Planner → workers → critic primitives | 🔜 in build |
| Trajectory evals, golden-run replay, CI gate, sabotage suite | 🔜 in build |
| Offline demo agent over grounded-mcp's vault | 🔜 in build |

Worth recording: the MCP bridge's **first integration test found a real bug in its sibling** — grounded-mcp v0.1's tools failed over real MCP transport (the SDK runs sync tools on worker threads; SQLite's same-thread default objected), invisible to grounded-mcp's own in-process eval suite. Fixed in grounded-mcp v0.1.1 the same hour. That is the argument for integration-level evals in one sentence.

## Quickstart (offline — no keys, no network)

```python
from grounded_harness import Agent, Budget, MockProvider, Tool, ToolCall, Turn
from grounded_harness.tools import Toolbox

lookup = Tool(
    name="lookup", description="Look a key up.",
    input_schema={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    fn=lambda key: {"price": "£249"}.get(key, f"no entry for {key}"),
)

script = [  # the deterministic mock provider consumes scripted turns
    Turn(tool_calls=[ToolCall(id="c1", name="lookup", input={"key": "price"})], stop_reason="tool_use"),
    Turn(text="The price is £249."),
]

run = Agent(MockProvider(script), Toolbox([lookup]), budget=Budget(max_steps=10)).run("What is the price?")
print(run.status, "->", run.final_text)   # completed -> The price is £249.
```

Swap `MockProvider` for `AnthropicProvider()` (needs `pip install "grounded-harness[anthropic]"` and `ANTHROPIC_API_KEY`) and nothing else changes — determinism for tests and replay, a live model for real runs, one seam.

## Agents over MCP servers

```python
from grounded_harness.tools import MCPServerSpec, mcp_toolbox

tools = mcp_toolbox(MCPServerSpec(
    command="grounded-mcp",                      # any stdio MCP server
    env={"GROUNDED_VAULT": "/path/to/vault"},
))
agent = Agent(AnthropicProvider(), tools)
```

The server's tools appear in the agent's toolbox with their real schemas; MCP-level errors surface as tool errors the agent (and the evals) can see, never as content that reads like an answer.

## Run discipline

- **Budgets end runs gracefully**: a `budget_exhausted` run is a valid outcome carrying every step taken and an honest partial summary, not an exception over a half-finished trajectory. Step ceilings stop spend *before* it happens; cost ceilings settle *after* the paid step is recorded, so no paid turn ever goes unrecorded.
- **Resume never re-bills**: completed steps replay from the atomic checkpoint. Resuming against *changed* inputs (task or toolset) is **refused** — a refused resume costs a re-run; a wrong resume costs correctness.

## Development

```bash
pip install -e ".[dev]"    # dev extra includes mcp + grounded-mcp for the bridge tests
pytest -q
```

## The grounded-* family

[grounded-mcp](https://github.com/rich-atkins/grounded-mcp) supplies knowledge with citations, abstention and entitlements; grounded-harness runs agents over it with budgets, checkpoints and (coming) trajectory evals. The two earlier starters — [agent-eval-starter](https://github.com/rich-atkins/agent-eval-starter) and [resumable-llm-pipeline](https://github.com/rich-atkins/resumable-llm-pipeline) — remain up as the documented origins of the eval-gate and resumability patterns this harness merges.

MIT © Richard Atkins
