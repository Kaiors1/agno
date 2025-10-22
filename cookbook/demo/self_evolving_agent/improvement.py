"""Structured models for the self-improvement workflow."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ToolPlan(BaseModel):
    """Instructions for enabling or disabling tools."""

    enable: List[str] = Field(default_factory=list)
    disable: List[str] = Field(default_factory=list)


class KnowledgeSourcePlan(BaseModel):
    """Actions to mutate the knowledge base."""

    name: str
    url: Optional[str] = None
    path: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class KnowledgePlan(BaseModel):
    """Aggregated knowledge actions."""

    refresh: List[str] = Field(default_factory=list)
    add: List[KnowledgeSourcePlan] = Field(default_factory=list)


class ImprovementSuggestion(BaseModel):
    """Structured response returned by the improvement agent."""

    summary: str
    instructions: Optional[str] = None
    tool_plan: ToolPlan = Field(default_factory=ToolPlan)
    knowledge_plan: KnowledgePlan = Field(default_factory=KnowledgePlan)
    follow_up_actions: List[str] = Field(default_factory=list)


def summarise_suggestion(suggestion: ImprovementSuggestion) -> str:
    """Render a compact human readable summary."""

    lines: List[str] = [f"Summary: {suggestion.summary}"]

    if suggestion.instructions:
        lines.append("- Updated instructions provided.")

    if suggestion.tool_plan.enable or suggestion.tool_plan.disable:
        if suggestion.tool_plan.enable:
            lines.append("- Enable tools: " + ", ".join(suggestion.tool_plan.enable))
        if suggestion.tool_plan.disable:
            lines.append("- Disable tools: " + ", ".join(suggestion.tool_plan.disable))

    if suggestion.knowledge_plan.refresh or suggestion.knowledge_plan.add:
        if suggestion.knowledge_plan.refresh:
            lines.append(
                "- Refresh knowledge sources: "
                + ", ".join(suggestion.knowledge_plan.refresh)
            )
        if suggestion.knowledge_plan.add:
            additions = ", ".join(src.name for src in suggestion.knowledge_plan.add)
            lines.append("- Add knowledge sources: " + additions)

    if suggestion.follow_up_actions:
        lines.append("- Follow-up actions:")
        lines.extend(f"  * {action}" for action in suggestion.follow_up_actions)

    return "\n".join(lines)
