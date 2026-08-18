"""Graph factory for LangGraph Studio (`uv run langgraph dev`).

Studio loads one importable graph per entry in `langgraph.json`. It is built without this
project's own persistence because the dev server supplies its own checkpointer and store; the
FastAPI app and CLI still build the same agent through `build_agent` with real Postgres/SQLite
persistence.
"""

from langchain_core.runnables import RunnableConfig

from app.agent.graph import build_agent
from app.agent.mcp_client import get_mcp_tools
from app.agent.registry import register_tools

_tools_registered = False


async def make_graph(config: RunnableConfig | None = None):
    """Build the agent for Studio, spawning the MCP server only on the first call.

    Async because MCP tools are resolved over stdio; Studio supports async factories directly.
    """
    global _tools_registered
    if not _tools_registered:
        register_tools(await get_mcp_tools())
        _tools_registered = True
    return build_agent(checkpointer=None, store=None)
