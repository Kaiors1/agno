# Self-Evolving Agno Agent

This demo ships an end-to-end project that showcases how to build an interactive
agent that keeps itself up to date with the Agno framework. The goal is to make
it easy to explore Agno's primitives while allowing the agent to analyse its
own configuration, extend its toolset and refresh its knowledge base directly
from the public documentation.

## Highlights

- **Interactive CLI** powered by [Typer](https://typer.tiangolo.com/) with
  conversational chat and self-improvement commands.
- **Knowledge orchestration** using the Agno `Knowledge` API backed by
  LanceDB for hybrid semantic search over the public docs and the local
  cookbook.
- **Self-improvement loop** implemented with structured outputs. The agent can
  recommend additional tools, update its instructions and request new knowledge
  sources, which are automatically persisted back into the configuration file.
- **Workspace aware tools** such as the Knowledge toolkit, Calculator, Python
  and Shell utilities that can be toggled directly from the improvement cycle.
- **Configuration-first design** – every part of the project is driven by a
  YAML config file so that teams can fork the template and adapt it to their
  stack in minutes.

## Project Layout

```
self_evolving_agent/
├── README.md
├── __init__.py
├── agent.py              # Runtime that loads configuration and instantiates Agno
├── catalog.py            # Registry of curated tools that can be toggled on demand
├── cli.py                # Typer application with chat / improve / refresh commands
├── config.py             # Pydantic models for loading & persisting YAML configs
├── config.yaml           # Default configuration shipped with the project
├── improvement.py        # Structured output models and utilities for self-updates
└── tests/
    ├── __init__.py
    └── test_config.py    # Example unit tests covering config round-trips
```

## Getting Started

1. **Install dependencies** (inside a virtual environment):

   ```bash
   pip install -r cookbook/demo/requirements.txt
   ```

2. **Set your model credentials**. The default configuration targets OpenAI, so
   export an `OPENAI_API_KEY`. You can switch to Anthropic or other providers by
   editing `config.yaml`.

3. **Launch the CLI**:

   ```bash
   python -m cookbook.demo.self_evolving_agent.cli chat
   ```

   Start chatting with `Agno AutoPilot`. Use `exit`, `quit` or `Ctrl+C` to stop
   the interactive session.

4. **Run the self-improvement cycle**:

   ```bash
   python -m cookbook.demo.self_evolving_agent.cli improve
   ```

   The agent will analyse its current toolset and instructions. Structured
   recommendations are automatically applied back into `config.yaml` and the
   runtime is reloaded with the new configuration.

5. **Refresh the knowledge base** whenever new docs are released:

   ```bash
   python -m cookbook.demo.self_evolving_agent.cli refresh-knowledge
   ```

   This command re-ingests all sources defined in `config.yaml`. You can pass
   `--source` multiple times to refresh specific entries only.

## Extending the Project

- Add new tools by registering them in `catalog.py`. The improvement agent will
  immediately become aware of the new capability and can opt-in when useful.
- Point the knowledge base to your private documentation or git repositories by
  editing `config.yaml`. LanceDB stores vectors locally under
  `tmp/self_evolving_agno_agent/lancedb` by default.
- Compose multi-agent teams or workflows by importing this agent into a larger
  application. The project is intentionally modular, so you can reuse the config
  loader and self-improvement utilities anywhere else in the Agno ecosystem.

## Running Tests

```
pytest cookbook/demo/self_evolving_agent/tests
```

This suite currently covers configuration loading and persistence logic. You
can expand it with integration tests that exercise the CLI or the improvement
loop using mock models.
