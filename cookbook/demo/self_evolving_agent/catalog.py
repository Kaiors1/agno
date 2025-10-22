"""Curated tool catalog for the self-evolving agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from agno.knowledge.knowledge import Knowledge
from agno.tools.calculator import CalculatorTools
from agno.tools.knowledge import KnowledgeTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools


@dataclass(frozen=True)
class ToolDescriptor:
    """Metadata describing an optional tool."""

    slug: str
    title: str
    description: str
    factory: Callable[["ToolContext"], object]
    requires_knowledge: bool = False


@dataclass
class ToolContext:
    """Dependencies passed to tool factories."""

    workspace_dir: Path
    knowledge: Optional[Knowledge]


def _knowledge_factory(ctx: ToolContext) -> KnowledgeTools:
    if ctx.knowledge is None:
        raise ValueError("KnowledgeTools require an active Knowledge instance")
    return KnowledgeTools(knowledge=ctx.knowledge)


def _python_factory(ctx: ToolContext) -> PythonTools:
    return PythonTools(base_dir=ctx.workspace_dir)


def _shell_factory(ctx: ToolContext) -> ShellTools:
    return ShellTools(base_dir=ctx.workspace_dir)


def _calculator_factory(ctx: ToolContext) -> CalculatorTools:
    return CalculatorTools()


TOOL_CATALOG: Dict[str, ToolDescriptor] = {
    "knowledge": ToolDescriptor(
        slug="knowledge",
        title="Knowledge Toolkit",
        description=(
            "Think/Search/Analyze tools for hybrid retrieval over the configured"
            " knowledge base."
        ),
        factory=_knowledge_factory,
        requires_knowledge=True,
    ),
    "python": ToolDescriptor(
        slug="python",
        title="Python Workspace",
        description="Execute Python snippets and manage local project files.",
        factory=_python_factory,
    ),
    "shell": ToolDescriptor(
        slug="shell",
        title="Shell Commands",
        description="Run shell commands within the agent workspace.",
        factory=_shell_factory,
    ),
    "calculator": ToolDescriptor(
        slug="calculator",
        title="Calculator",
        description="Perform reliable arithmetic for planning and estimations.",
        factory=_calculator_factory,
    ),
}


def instantiate_tools(
    selected: Iterable[str],
    *,
    context: ToolContext,
    allow_missing: bool = False,
) -> List[object]:
    """Instantiate toolkits based on their slugs."""

    instances: List[object] = []
    for slug in selected:
        descriptor = TOOL_CATALOG.get(slug)
        if descriptor is None:
            if allow_missing:
                continue
            raise KeyError(f"Unknown tool slug '{slug}'. Register it in TOOL_CATALOG.")

        if descriptor.requires_knowledge and context.knowledge is None:
            raise RuntimeError(
                f"Tool '{slug}' requires a knowledge base but none is configured."
            )

        instances.append(descriptor.factory(context))
    return instances


def describe_catalog() -> List[Dict[str, str]]:
    """Return a lightweight description of the full catalog."""

    return [
        {
            "slug": descriptor.slug,
            "title": descriptor.title,
            "description": descriptor.description,
        }
        for descriptor in TOOL_CATALOG.values()
    ]
