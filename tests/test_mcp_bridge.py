"""Integration: the MCP bridge against a real grounded-mcp server (stdio, offline).

This is the composition the harness exists for: an agent whose knowledge tool is
an MCP server. Runs the actual grounded-mcp console script from this venv over
its ACME demo vault — no network, no credentials.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from grounded_harness.tools import MCPServerSpec, mcp_toolbox

VAULT = Path(__file__).parent.parent.parent / "grounded-mcp" / "demo_vault"
SERVER = Path(sys.executable).parent / "grounded-mcp"

pytestmark = pytest.mark.skipif(
    not (SERVER.exists() and VAULT.exists()),
    reason="grounded-mcp not installed alongside (dev extra)",
)


def _spec() -> MCPServerSpec:
    env = dict(os.environ)
    env["GROUNDED_VAULT"] = str(VAULT)
    env["GROUNDED_PROFILE"] = "staff"
    return MCPServerSpec(command=str(SERVER), env=env)


def test_bridge_lists_the_servers_tools():
    box = mcp_toolbox(_spec())
    names = box.names()
    assert {"search", "read_note", "backlinks", "browse"} <= set(names)


def test_bridge_proxies_a_search_call():
    from grounded_harness.providers import ToolCall
    box = mcp_toolbox(_spec())
    result = box.run(ToolCall(id="t1", name="search",
                              input={"query": "widget pro pricing", "k": 2}))
    assert result.is_error is False, result.content
    payload = json.loads(result.content)
    assert payload["abstained"] is False
    assert payload["hits"][0]["citation_id"].startswith("public/products/widget-pro.md")
