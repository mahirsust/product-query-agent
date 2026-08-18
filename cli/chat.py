"""Development REPL for the agent.

A convenience wrapper around the same agent the API serves, bypassing HTTP and auth. It is not a
second production surface; there is no user identity, so per-user usage tracking and long-term
memory are inactive here.
"""

import asyncio

from dotenv import load_dotenv

from app.agent.graph import build_agent
from app.agent.mcp_client import get_mcp_tools
from app.agent.registry import register_tools
from app.agent.store import get_store
from app.db.checkpointer import get_checkpointer
from app.logging_config import configure_logging, configure_tracing, set_correlation_id

_THREAD_ID = "cli-session"
_EXIT_COMMANDS = {"exit", "quit"}


async def ask(agent, question: str, thread_id: str) -> str:
    """Ask the agent a question within a conversation thread and return its final answer."""
    set_correlation_id(thread_id)
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return response["messages"][-1].content


async def _run_repl() -> None:
    """Resolve tools, build the agent, and loop on stdin until the user exits.

    Async throughout because MCP-sourced tools implement only the async interface; a synchronous
    `invoke` raises as soon as one is called. Blocking `input()` is fine inside it — a single-user
    REPL has nothing to run concurrently.
    """
    register_tools(await get_mcp_tools())

    print("Product Query Agent. Type 'exit' to quit.")
    async with get_checkpointer() as checkpointer, get_store() as store:
        agent = build_agent(checkpointer=checkpointer, store=store)
        while True:
            question = input("You: ").strip()
            if question.lower() in _EXIT_COMMANDS:
                break
            if not question:
                continue
            print("Agent:", await ask(agent, question, _THREAD_ID))


def main() -> None:
    """Entry point for `uv run main.py`."""
    load_dotenv()
    configure_logging()
    configure_tracing()
    asyncio.run(_run_repl())


if __name__ == "__main__":
    main()
