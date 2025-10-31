# Flowchart Agent

Flowchart Agent takes natural language, code, or other text input and produces Mermaid code along with a rendered HTML flowchart. It uses LangChain with a self-reflection loop to refine its output until the diagram meets quality bar or reaches the iteration limit.

## Requirements

- Python 3.10+
- `uv` package manager
- An OpenAI-compatible API key exported as `OPENAI_API_KEY`

## Setup

```bash
uv sync
```

## Run

```bash
uv run flowchart-agent "Describe the onboarding process"
```

The command prints Mermaid source and writes `flowchart_output.html` with rendered diagram.
