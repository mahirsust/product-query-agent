"""Measure per-query cost drivers: LLM calls, tool calls, tokens, and latency.

Run before and after changing tools, prompts, or middleware to see the effect:
    uv run python -m scripts.benchmark
"""

import asyncio
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agent.graph import build_agent
from app.agent.mcp_client import get_mcp_tools
from app.agent.registry import get_all_tools, register_tools
from app.config import settings

QUERIES = [
    "what is the price of the macbook?",
    "what is the price of the macbook and what are its reviews?",
    "what fragrances do you have under 100 dollars?",
    "what products do you have",
]


async def measure(agent, question: str, thread_id: str) -> dict:
    """Run one question and report its cost drivers."""
    started = time.perf_counter()
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.max_recursion_limit,
        },
        context={"user_id": 0},
    )
    elapsed = time.perf_counter() - started

    messages = response["messages"]
    tool_calls = [m.name for m in messages if isinstance(m, ToolMessage) and m.name]
    llm_calls = [m for m in messages if isinstance(m, AIMessage)]
    tokens = sum((m.usage_metadata or {}).get("total_tokens", 0) for m in llm_calls)

    return {
        "question": question,
        "llm_calls": len(llm_calls),
        "tool_calls": tool_calls,
        "tokens": tokens,
        "seconds": round(elapsed, 1),
        "answer": str(messages[-1].content)[:70],
    }


async def main() -> None:
    """Benchmark every query and print per-question and total figures."""
    load_dotenv()
    settings.cost_budget_usd_per_user_per_day = 100.0
    settings.max_llm_calls_per_minute_per_user = 1000

    register_tools(await get_mcp_tools())
    agent = build_agent(checkpointer=InMemorySaver(), store=InMemoryStore())

    print(f"model={settings.llm_model}  tools={[t.name for t in get_all_tools()]}\n")

    totals = {"llm_calls": 0, "tokens": 0, "tool_calls": 0, "seconds": 0.0}
    for i, question in enumerate(QUERIES):
        result = await measure(agent, question, thread_id=f"bench-{i}")
        totals["llm_calls"] += result["llm_calls"]
        totals["tokens"] += result["tokens"]
        totals["tool_calls"] += len(result["tool_calls"])
        totals["seconds"] += result["seconds"]
        print(f"Q: {result['question']}")
        print(
            f"   llm_calls={result['llm_calls']}  tools={result['tool_calls']}  "
            f"tokens={result['tokens']}  {result['seconds']}s"
        )
        print(f"   -> {result['answer']}\n")

    print(
        f"TOTAL  llm_calls={totals['llm_calls']}  tool_calls={totals['tool_calls']}  "
        f"tokens={totals['tokens']}  {round(totals['seconds'], 1)}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
