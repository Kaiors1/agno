"""Configuration models and helpers for the self-evolving Agno agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Model provider configuration."""

    provider: Literal["openai", "anthropic", "groq", "google", "nebius"]
    id: str = Field(..., description="Model identifier exposed by the provider.")
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class EmbedderConfig(BaseModel):
    """Embedding model configuration."""

    provider: Literal["openai", "fastembed"] = "openai"
    model: str = "text-embedding-3-small"


class KnowledgeSourceConfig(BaseModel):
    """Single knowledge source definition."""

    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_location(self) -> "KnowledgeSourceConfig":
        if not self.url and not self.path:
            raise ValueError("Each knowledge source must define a `url` or `path`.")
        return self


class KnowledgeConfig(BaseModel):
    """Knowledge base settings."""

    enabled: bool = True
    vector_store_uri: str
    table_name: str = "agno_docs"
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    sources: List[KnowledgeSourceConfig] = Field(default_factory=list)


class AgentProfile(BaseModel):
    """Describes the public persona of the agent."""

    name: str = "Agno AutoPilot"
    description: str
    instructions: str


class ToolConfig(BaseModel):
    """Controls which tools are enabled."""

    enabled: List[str] = Field(default_factory=list)
    optional: List[str] = Field(default_factory=list)

    @field_validator("enabled", "optional", mode="before")
    @classmethod
    def _normalise_entries(cls, value: Optional[Iterable[str]]) -> List[str]:
        if value is None:
            return []
        return [str(v).strip() for v in value if str(v).strip()]


class SelfImprovementConfig(BaseModel):
    """Parameters used by the self-improvement loop."""

    enabled: bool = True
    goal: str = Field(
        default=(
            "Continuously expand the agent's knowledge of Agno, refresh tooling and"
            " deliver production-ready guidance to developers."
        )
    )
    max_tool_changes: int = Field(1, ge=0, le=10)
    allow_disabling_tools: bool = True
    refresh_on_startup: bool = True


class AgentConfig(BaseModel):
    """Root configuration model."""

    agent: AgentProfile
    model: ModelConfig
    knowledge: KnowledgeConfig
    tools: ToolConfig = Field(default_factory=ToolConfig)
    self_improvement: SelfImprovementConfig = Field(default_factory=SelfImprovementConfig)

    def save(self, path: Path) -> None:
        """Persist the configuration back to disk."""

        payload = yaml.safe_dump(self.model_dump(mode="python"), sort_keys=False)
        path.write_text(payload, encoding="utf-8")


@dataclass(slots=True)
class LoadedConfig:
    """Utility wrapper bundling config and paths."""

    config: AgentConfig
    path: Path
    workspace_dir: Path


def load_config(path: Path) -> LoadedConfig:
    """Load a configuration file and resolve the workspace path."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = AgentConfig.model_validate(raw_data)

    workspace_dir = (path.parent / "workspace").resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    return LoadedConfig(config=config, path=path, workspace_dir=workspace_dir)
