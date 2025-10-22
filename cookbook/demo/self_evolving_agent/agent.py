"""Runtime utilities for the self-evolving Agno agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Sequence

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.models.nebius import Nebius
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb, SearchType

from .catalog import ToolContext, describe_catalog, instantiate_tools
from .config import AgentConfig, KnowledgeSourceConfig, LoadedConfig, load_config
from .improvement import ImprovementSuggestion


@dataclass(slots=True)
class AgentRuntime:
    """Holds instantiated runtime objects."""

    agent: Agent
    improvement_agent: Agent
    knowledge: Optional[Knowledge]
    db: SqliteDb


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _create_model(config: AgentConfig) -> object:
    provider = config.model.provider
    if provider == "openai":
        return OpenAIChat(id=config.model.id, temperature=config.model.temperature)
    if provider == "anthropic":
        return Claude(id=config.model.id, temperature=config.model.temperature)
    if provider == "groq":
        return Groq(id=config.model.id, temperature=config.model.temperature)
    if provider == "google":
        return Gemini(id=config.model.id, temperature=config.model.temperature)
    if provider == "nebius":
        return Nebius(id=config.model.id, temperature=config.model.temperature)
    raise ValueError(f"Unsupported model provider '{provider}'")


def _create_embedder(config: AgentConfig) -> object:
    embedder_cfg = config.knowledge.embedder
    if embedder_cfg.provider == "openai":
        return OpenAIEmbedder(id=embedder_cfg.model)
    if embedder_cfg.provider == "fastembed":
        return FastEmbedEmbedder(model_name=embedder_cfg.model)
    raise ValueError(f"Unsupported embedder provider '{embedder_cfg.provider}'")


class SelfEvolvingAgnoAgent:
    """Facade responsible for managing the runtime and self-improvement loop."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        default_path = Path(__file__).with_name("config.yaml")
        path = config_path or default_path
        self.loaded: LoadedConfig = load_config(path)
        self.config: AgentConfig = self.loaded.config
        self.runtime: AgentRuntime = self._build_runtime()

        if self.config.self_improvement.refresh_on_startup and self.runtime.knowledge:
            self.refresh_knowledge()

    @property
    def config_path(self) -> Path:
        return self.loaded.path

    @property
    def workspace_dir(self) -> Path:
        return self.loaded.workspace_dir

    # ------------------------------------------------------------------
    # Runtime builders
    # ------------------------------------------------------------------
    def _build_runtime(self) -> AgentRuntime:
        model = _create_model(self.config)
        db_path = self.loaded.workspace_dir / "agent_state.db"
        db = SqliteDb(db_file=str(db_path))

        knowledge = None
        if self.config.knowledge.enabled:
            embedder = _create_embedder(self.config)
            vector_uri = _resolve_path(self.loaded.path.parent, self.config.knowledge.vector_store_uri)
            vector_uri.parent.mkdir(parents=True, exist_ok=True)
            knowledge = Knowledge(
                name="Agno Framework Docs",
                description="Aggregated documentation sources for the Agno framework.",
                vector_db=LanceDb(
                    uri=str(vector_uri),
                    table_name=self.config.knowledge.table_name,
                    search_type=SearchType.hybrid,
                    embedder=embedder,
                ),
                contents_db=db,
            )

        tool_context = ToolContext(workspace_dir=self.loaded.workspace_dir, knowledge=knowledge)
        tools = instantiate_tools(self.config.tools.enabled, context=tool_context, allow_missing=True)

        agent = Agent(
            name=self.config.agent.name,
            description=self.config.agent.description,
            instructions=self.config.agent.instructions,
            model=model,
            db=db,
            knowledge=knowledge,
            tools=tools,
            search_knowledge=self.config.knowledge.enabled,
            add_history_to_context=True,
            add_datetime_to_context=True,
            markdown=True,
            telemetry=False,
        )

        improvement_agent = self._build_improvement_agent(model, knowledge, db)

        return AgentRuntime(agent=agent, improvement_agent=improvement_agent, knowledge=knowledge, db=db)

    def _build_improvement_agent(
        self, model: object, knowledge: Optional[Knowledge], db: SqliteDb
    ) -> Agent:
        catalog_rows = describe_catalog()
        catalog_text = "\n".join(
            f"- {row['slug']}: {row['title']} — {row['description']}" for row in catalog_rows
        )
        enabled_tools = ", ".join(self.config.tools.enabled) or "none"
        optional_tools = ", ".join(self.config.tools.optional) or "none"
        knowledge_rows = []
        for source in self.config.knowledge.sources:
            location = source.url or source.path or "<missing>"
            knowledge_rows.append(f"- {source.name}: {location}")
        knowledge_text = "\n".join(knowledge_rows) if knowledge_rows else "- No sources configured"

        instructions = dedent(
            f"""
            You are the improvement coordinator for {self.config.agent.name}. Analyse
            the agent's current configuration and propose actionable upgrades that
            keep it aligned with the latest Agno capabilities.

            Always respond with a valid JSON object that conforms to the
            `ImprovementSuggestion` schema.

            ## Current configuration
            - Enabled tools: {enabled_tools}
            - Optional tools: {optional_tools}

            ## Tool catalog
            {catalog_text}

            ## Knowledge sources
            {knowledge_text}

            Include concrete descriptions and URLs when suggesting new knowledge
            sources. When you disable a tool explain the rationale in the
            `summary` field. Respect the `max_tool_changes` limit and prefer
            high-impact upgrades.
            """
        ).strip()

        tool_context = ToolContext(workspace_dir=self.loaded.workspace_dir, knowledge=knowledge)
        improvement_tools: List[object] = []
        if knowledge is not None:
            improvement_tools = instantiate_tools(["knowledge"], context=tool_context, allow_missing=True)

        return Agent(
            name=f"{self.config.agent.name} - Improvement",
            description="Coordinates self-improvement cycles.",
            instructions=instructions,
            model=model,
            db=db,
            knowledge=knowledge,
            tools=improvement_tools,
            search_knowledge=knowledge is not None,
            markdown=False,
            telemetry=False,
            output_schema=ImprovementSuggestion,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(self, message: str, *, stream: bool = False):
        if stream:
            return self.runtime.agent.run(message, stream=True)
        return self.runtime.agent.run(message)

    def run_self_improvement(self, goal: Optional[str] = None) -> ImprovementSuggestion:
        if not self.config.self_improvement.enabled:
            raise RuntimeError("Self-improvement is disabled in the configuration.")

        prompt = self._build_improvement_prompt(goal)
        response = self.runtime.improvement_agent.run(prompt)
        suggestion = response.content
        if not isinstance(suggestion, ImprovementSuggestion):
            raise ValueError("Improvement agent returned an unexpected payload.")

        self._apply_suggestion(suggestion)
        return suggestion

    def refresh_knowledge(
        self, *, source_names: Optional[Sequence[str]] = None, force: bool = False
    ) -> None:
        if self.runtime.knowledge is None:
            return

        skip_if_exists = not force
        for source in self.config.knowledge.sources:
            if source_names and source.name not in source_names:
                continue

            kwargs = {
                "name": source.name,
                "description": source.description,
                "metadata": source.metadata or None,
                "include": source.include or None,
                "exclude": source.exclude or None,
                "skip_if_exists": skip_if_exists,
                "upsert": True,
            }
            if source.url:
                self.runtime.knowledge.add_content(url=source.url, **kwargs)
            if source.path:
                resolved = _resolve_path(self.loaded.path.parent, source.path)
                self.runtime.knowledge.add_content(path=str(resolved), **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_improvement_prompt(self, goal: Optional[str]) -> str:
        goal_text = goal or self.config.self_improvement.goal
        return dedent(
            f"""
            Evaluate the current configuration and recommend upgrades that keep the
            agent aligned with Agno best practices.

            Goal: {goal_text}

            Remember to stick to the ImprovementSuggestion schema and keep the
            proposed changes within the allowed limits.
            """
        ).strip()

    def _apply_suggestion(self, suggestion: ImprovementSuggestion) -> None:
        changes_applied = False

        if suggestion.instructions:
            self.config.agent.instructions = suggestion.instructions
            changes_applied = True

        enabled = list(self.config.tools.enabled)
        max_changes = self.config.self_improvement.max_tool_changes
        additions = 0
        for slug in suggestion.tool_plan.enable:
            if slug not in enabled and additions < max_changes:
                enabled.append(slug)
                additions += 1
                changes_applied = True
        if self.config.self_improvement.allow_disabling_tools:
            for slug in suggestion.tool_plan.disable:
                if slug in enabled:
                    enabled.remove(slug)
                    changes_applied = True
        self.config.tools.enabled = enabled

        for source_plan in suggestion.knowledge_plan.add:
            if not any(src.name == source_plan.name for src in self.config.knowledge.sources):
                if not (source_plan.url or source_plan.path):
                    continue
                new_source = KnowledgeSourceConfig(
                    name=source_plan.name,
                    description=source_plan.description,
                    url=source_plan.url,
                    path=source_plan.path,
                    metadata=source_plan.metadata,
                )
                self.config.knowledge.sources.append(new_source)
                self.refresh_knowledge(source_names=[source_plan.name], force=True)
                changes_applied = True

        if suggestion.knowledge_plan.refresh:
            self.refresh_knowledge(source_names=suggestion.knowledge_plan.refresh, force=True)

        if changes_applied:
            self.config.save(self.loaded.path)
            self.runtime = self._build_runtime()

    # ------------------------------------------------------------------
    # Utility helpers for CLI
    # ------------------------------------------------------------------
    def list_enabled_tools(self) -> List[str]:
        return list(self.config.tools.enabled)
