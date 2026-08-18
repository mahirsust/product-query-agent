"""Chat endpoint: the authenticated entry point to the agent."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from app.api.deps import get_agent, get_current_user
from app.api.schemas import ChatRequest, ChatResponse
from app.cache import response_cache, usage_tracker
from app.cache.usage_tracker import UsageCheckResult
from app.config import settings
from app.db.models import User
from app.logging_config import set_correlation_id
from app.security.rate_limit import limiter

router = APIRouter(tags=["chat"])


def _namespaced_thread_id(user_id: int, thread_id: str) -> str:
    """Scope a client-supplied thread id to its owner.

    Applied before the id reaches the checkpointer so two users cannot collide on the same thread
    string and read or overwrite each other's history.
    """
    return f"user-{user_id}:{thread_id}"


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    payload: ChatRequest,
    agent=Depends(get_agent),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """Answer a question, subject to caching, rate limits, and the per-user budget.

    Returns the answer plus the tools used on this turn. Rejects with 429 when a per-minute rate
    is exceeded and 402 when a daily cap is exhausted — distinct so the caller knows whether
    retrying soon will help.
    """
    set_correlation_id(_namespaced_thread_id(current_user.id, payload.thread_id))
    config = {
        "configurable": {"thread_id": _namespaced_thread_id(current_user.id, payload.thread_id)},
        "recursion_limit": settings.max_recursion_limit,
        "tags": [f"user:{current_user.id}"],
        "metadata": {"user_id": current_user.id, "thread_id": payload.thread_id},
    }

    prior_state = await agent.aget_state(config)
    prior_message_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0
    is_first_turn = prior_message_count == 0

    # Only first turns are cacheable: a follow-up's answer depends on conversation context that
    # the question text alone does not capture. Checked before the usage gate, since serving a
    # cached answer costs nothing.
    if is_first_turn:
        cached_answer = await response_cache.get(payload.question, current_user.id)
        if cached_answer is not None:
            return ChatResponse(answer=cached_answer, thread_id=payload.thread_id, tool_calls=[])

    usage_check = await usage_tracker.check_and_reserve(current_user.id)
    if usage_check == UsageCheckResult.RATE_LIMITED:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests — slow down and try again in a moment.",
        )
    if usage_check == UsageCheckResult.BUDGET_EXCEEDED:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Daily usage budget exhausted for this account.",
        )

    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": payload.question}]},
            config=config,
            context={"user_id": current_user.id},
        )
    except GraphRecursionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not answer this within the agent's step limit. Try a simpler question.",
        ) from exc

    # Report only this turn's tool calls, not the thread's whole history.
    new_messages = response["messages"][prior_message_count:]
    tool_calls = [m.name for m in new_messages if isinstance(m, ToolMessage) and m.name]
    answer = response["messages"][-1].content

    if is_first_turn:
        await response_cache.set(payload.question, answer, current_user.id)

    return ChatResponse(answer=answer, thread_id=payload.thread_id, tool_calls=tool_calls)
