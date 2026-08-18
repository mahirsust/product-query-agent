from langchain_core.tools import tool

from app.agent.registry import (
    get_all_tools,
    get_sensitive_tools,
    register_tool,
    register_tools,
)


@tool
def sample_tool_a(x: int) -> int:
    """A sample tool."""
    return x


@tool
def sample_tool_b(x: int) -> int:
    """Another sample tool."""
    return x


def test_register_tool_default_not_sensitive():
    register_tool(sample_tool_a)
    assert sample_tool_a in get_all_tools()
    assert sample_tool_a not in get_sensitive_tools()


def test_register_tool_sensitive_flag():
    register_tool(sample_tool_b, sensitive=True)
    assert sample_tool_b in get_all_tools()
    assert sample_tool_b in get_sensitive_tools()


def test_register_tool_decorator_bare():
    @register_tool
    @tool
    def bare_tool(x: int) -> int:
        """Bare decorator usage."""
        return x

    assert bare_tool in get_all_tools()
    assert bare_tool not in get_sensitive_tools()


def test_register_tool_decorator_parameterized():
    @register_tool(sensitive=True)
    @tool
    def parameterized_tool(x: int) -> int:
        """Parameterized decorator usage."""
        return x

    assert parameterized_tool in get_sensitive_tools()


def test_register_tools_bulk_defaults_not_sensitive():
    register_tools([sample_tool_a, sample_tool_b])
    all_tools = get_all_tools()
    assert sample_tool_a in all_tools
    assert sample_tool_b in all_tools
    assert get_sensitive_tools() == []


def test_register_tool_overwrites_by_name_not_duplicates():
    register_tool(sample_tool_a)
    assert len(get_all_tools()) == 1
    register_tool(sample_tool_a, sensitive=True)
    assert len(get_all_tools()) == 1
    assert sample_tool_a in get_sensitive_tools()


def test_empty_registry_returns_empty_lists():
    assert get_all_tools() == []
    assert get_sensitive_tools() == []
