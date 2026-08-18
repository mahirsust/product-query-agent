"""MCP client that resolves the catalogue tools from the local MCP server."""

import os
import sys

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_mcp_tools() -> list[BaseTool]:
    """Resolve the catalogue tools by spawning the MCP server as a subprocess.

    Intended to run once at startup rather than per request.
    """
    client = MultiServerMCPClient(
        {
            "dummyjson": {
                "transport": "stdio",
                # sys.executable rather than "python": guarantees the subprocess runs in this
                # same interpreter and virtualenv.
                "command": sys.executable,
                "args": ["-m", "mcp_servers.dummyjson.server"],
                # The stdio transport passes a minimal environment unless one is given, which
                # left the server without DATABASE_URL and silently falling back to the SQLite
                # default. It runs our own code and shares this app's configuration, so it gets
                # the same environment as the parent.
                "env": dict(os.environ),
            }
        }
    )
    return await client.get_tools()
