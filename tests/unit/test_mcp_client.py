"""Tests for how the MCP server subprocess is launched."""

import asyncio
import os
from unittest.mock import patch

from app.agent import mcp_client


def test_subprocess_receives_the_parent_environment():
    """The stdio transport passes a minimal environment unless one is supplied. Without this the
    server loses DATABASE_URL and silently falls back to the SQLite default, which in a container
    running as non-root fails outright with 'unable to open database file'."""
    captured = {}

    class _FakeClient:
        def __init__(self, connections):
            captured.update(connections)

        async def get_tools(self):
            return []

    with patch.object(mcp_client, "MultiServerMCPClient", _FakeClient):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql+psycopg://u:p@h/db"}):
            asyncio.run(mcp_client.get_mcp_tools())

    env = captured["dummyjson"]["env"]
    assert env["DATABASE_URL"] == "postgresql+psycopg://u:p@h/db"
