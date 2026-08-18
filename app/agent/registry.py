"""Tool registry.

Tools self-declare a `sensitive` flag here instead of being hand-listed at the call site, which
is what lets the HITL middleware gate a future write tool without plumbing changes.
"""

from dataclasses import dataclass

from langchain_core.tools import BaseTool


@dataclass
class ToolSpec:
    """A registered tool and whether it needs human approval before running."""

    name: str
    description: str
    fn: BaseTool
    sensitive: bool = False


_registry: dict[str, ToolSpec] = {}


def register_tool(fn: BaseTool | None = None, *, sensitive: bool = False):
    """Register a single tool. Usable bare (`@register_tool`) or parameterized
    (`@register_tool(sensitive=True)`) for tools defined locally with `@tool`."""

    def _register(tool: BaseTool) -> BaseTool:
        """Record the tool, replacing any earlier registration under the same name."""
        _registry[tool.name] = ToolSpec(
            name=tool.name, description=tool.description, fn=tool, sensitive=sensitive
        )
        return tool

    if fn is not None:
        return _register(fn)
    return _register


def register_tools(tools: list[BaseTool], *, sensitive: bool = False) -> None:
    """Bulk-register tools resolved dynamically (e.g. from an MCP client), which can't use the
    `@register_tool` decorator since they aren't defined as local functions."""
    for tool in tools:
        register_tool(tool, sensitive=sensitive)


def get_all_tools() -> list[BaseTool]:
    """Every registered tool, in registration order."""
    return [spec.fn for spec in _registry.values()]


def get_sensitive_tools() -> list[BaseTool]:
    """Only tools requiring approval. Empty today: every current tool is read-only."""
    return [spec.fn for spec in _registry.values() if spec.sensitive]
