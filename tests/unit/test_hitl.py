from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool

from app.agent.middleware.hitl import get_hitl_middleware
from app.agent.registry import register_tool


@tool
def safe_tool(x: int) -> int:
    """Safe read-only tool."""
    return x


@tool
def dangerous_tool(x: int) -> int:
    """Sensitive mutating tool."""
    return x


def test_no_sensitive_tools_omits_the_middleware():
    """Returns None rather than an inert middleware, so the stack doesn't pay graph steps for a
    hook that can never fire."""
    register_tool(safe_tool, sensitive=False)
    assert get_hitl_middleware() is None


def test_gates_only_sensitive_tools():
    register_tool(safe_tool, sensitive=False)
    register_tool(dangerous_tool, sensitive=True)
    mw = get_hitl_middleware()
    assert set(mw.interrupt_on.keys()) == {"dangerous_tool"}


def test_empty_registry_omits_the_middleware():
    assert get_hitl_middleware() is None


def test_middleware_is_included_once_a_tool_is_sensitive():
    register_tool(dangerous_tool, sensitive=True)
    mw = get_hitl_middleware()
    assert isinstance(mw, HumanInTheLoopMiddleware)
    assert set(mw.interrupt_on) == {"dangerous_tool"}
