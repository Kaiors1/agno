from pathlib import Path

import pytest

from cookbook.demo.self_evolving_agent import config as agent_config
from cookbook.demo.self_evolving_agent.catalog import ToolContext, instantiate_tools


def test_config_round_trip(tmp_path: Path) -> None:
    source_config = Path(__file__).resolve().parent.parent / "config.yaml"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")

    loaded = agent_config.load_config(config_path)
    assert loaded.config.agent.name == "Agno AutoPilot"

    loaded.config.agent.name = "Test Agent"
    loaded.config.save(config_path)

    reloaded = agent_config.load_config(config_path)
    assert reloaded.config.agent.name == "Test Agent"


def test_knowledge_tool_requires_context(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_dir=tmp_path, knowledge=None)
    with pytest.raises(RuntimeError):
        instantiate_tools(["knowledge"], context=ctx)
