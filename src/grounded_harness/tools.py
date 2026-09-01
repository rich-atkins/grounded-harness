"""Tools: plain callables with schemas, plus an MCP bridge.

A tool failure is DATA, not an exception: the agent sees the error text and gets to
recover, and the trajectory eval sees it too. Only harness bugs raise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable


class ToolError(Exception):
    """Raised by the harness for tool-registry misuse (unknown tool, bad schema)."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., object]

    def spec(self) -> dict:
        """Anthropic-wire tool specification."""
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class Toolbox:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def run(self, call) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.id, call.name,
                              f"unknown tool: {call.name}", is_error=True)
        try:
            out = tool.fn(**call.input)
        except Exception as e:  # tool failures are data the agent can react to
            return ToolResult(call.id, call.name, f"tool error: {e}", is_error=True)
        content = out if isinstance(out, str) else json.dumps(out, default=str)
        return ToolResult(call.id, call.name, content)


@dataclass
class MCPServerSpec:
    """How to launch an MCP server over stdio (e.g. grounded-mcp)."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def mcp_toolbox(spec: MCPServerSpec) -> "MCPToolbox":
    """Toolbox whose tools proxy to an MCP server over stdio (lazy `mcp` import)."""
    return MCPToolbox(spec)


class MCPToolbox(Toolbox):
    """Bridges an MCP server's tools into the harness.

    The connection is opened per call-session via a context manager because MCP
    stdio clients are async and the harness loop is sync; `run_sync` wraps the
    async client. Verified against the installed mcp SDK by the test suite (the
    bridge is covered by an integration test against grounded-mcp's demo vault).
    """

    def __init__(self, spec: MCPServerSpec):
        super().__init__()
        self._spec = spec
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        for name, description, schema in self._list_remote_tools():
            remote = name

            def _proxy(_remote=remote, **kwargs):
                content, is_error = self._call_remote(_remote, kwargs)
                if is_error:
                    raise RuntimeError(content or f"MCP tool {_remote} errored")
                return content

            self.add(Tool(name=name, description=description,
                          input_schema=schema, fn=_proxy))
        self._loaded = True

    def specs(self) -> list[dict]:
        self._ensure_loaded()
        return super().specs()

    def names(self) -> list[str]:
        self._ensure_loaded()
        return super().names()

    def run(self, call):
        self._ensure_loaded()
        return super().run(call)

    # -- mcp SDK plumbing (async under a sync wrapper) -------------------------

    def _run_async(self, coro):
        import asyncio
        return asyncio.run(coro)

    async def _session(self):
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        params = StdioServerParameters(command=self._spec.command,
                                       args=self._spec.args, env=self._spec.env)
        return stdio_client(params), ClientSession

    def _list_remote_tools(self) -> list[tuple[str, str, dict]]:
        async def go():
            transport, ClientSession = await self._session()
            async with transport as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    return [(t.name, t.description or "", t.input_schema or {})
                            for t in listed.tools]
        return self._run_async(go())

    def _call_remote(self, name: str, arguments: dict) -> tuple[str, bool]:
        async def go():
            transport, ClientSession = await self._session()
            async with transport as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    parts = []
                    for item in result.content:
                        text = getattr(item, "text", None)
                        if text is not None:
                            parts.append(text)
                    # The server's error flag must survive the bridge: an MCP
                    # tool error rendered as ordinary content would read as a
                    # successful answer to both the agent and the evals.
                    is_error = bool(getattr(result, "is_error", False)
                                    or getattr(result, "isError", False))
                    return "\n".join(parts), is_error
        return self._run_async(go())
