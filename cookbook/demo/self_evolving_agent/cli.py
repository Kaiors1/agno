"""Typer CLI exposing the self-evolving Agno agent."""

from __future__ import annotations

import json
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .agent import SelfEvolvingAgnoAgent
from .improvement import summarise_suggestion

app = typer.Typer(help="Interact with the self-evolving Agno agent.")
console = Console()


def _print_response(response) -> None:
    content = getattr(response, "content", None) or "(no content)"
    console.print(Panel.fit(str(content), title="Agno AutoPilot", border_style="green"))


@app.command()
def chat(prompt: Optional[str] = typer.Argument(None), stream: bool = typer.Option(False, help="Stream the response.")) -> None:
    """Start a chat session with Agno AutoPilot."""

    agent = SelfEvolvingAgnoAgent()

    if prompt is not None:
        response = agent.chat(prompt, stream=stream)
        if stream:
            for chunk in response:
                if getattr(chunk, "content", None):
                    console.print(str(chunk.content))
        else:
            _print_response(response)
        return

    console.print("[bold cyan]Interactive session started. Type 'exit' or 'quit' to leave.[/bold cyan]")
    while True:
        try:
            user_input = console.input("[bold blue]You[/bold blue]: ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye 👋")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("Goodbye 👋")
            break

        response = agent.chat(user_input)
        _print_response(response)


@app.command()
def improve(goal: Optional[str] = typer.Option(None, help="Override the improvement goal for this run.")) -> None:
    """Run a self-improvement cycle and persist the results."""

    agent = SelfEvolvingAgnoAgent()
    suggestion = agent.run_self_improvement(goal=goal)

    console.print(Panel.fit(summarise_suggestion(suggestion), title="Improvement Summary", border_style="yellow"))
    console.print(json.dumps(suggestion.model_dump(mode="json"), indent=2))


@app.command(name="refresh-knowledge")
def refresh_knowledge(
    source: Optional[List[str]] = typer.Option(None, "--source", "-s", help="Limit refresh to specific sources."),
    force: bool = typer.Option(False, help="Re-ingest content even if it already exists."),
) -> None:
    """Re-ingest knowledge sources defined in the configuration."""

    agent = SelfEvolvingAgnoAgent()
    agent.refresh_knowledge(source_names=source, force=force)
    console.print("[green]Knowledge refresh complete.[/green]")


@app.command()
def status() -> None:
    """Show the current configuration snapshot."""

    agent = SelfEvolvingAgnoAgent()
    enabled = ", ".join(agent.list_enabled_tools()) or "none"
    sources = ", ".join(src.name for src in agent.config.knowledge.sources) or "none"

    console.print(
        Panel.fit(
            f"Config: {agent.config_path}\n"
            f"Workspace: {agent.workspace_dir}\n"
            f"Model: {agent.config.model.provider} / {agent.config.model.id}\n"
            f"Enabled tools: {enabled}\n"
            f"Knowledge sources: {sources}",
            title="Agno AutoPilot Status",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    app()
