from __future__ import annotations

from pathlib import Path

import typer

from .agent import FlowchartAgent, FlowchartAgentConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    prompt: str = typer.Argument(
        ...,
        help="Natural language, code, or other text to turn into a flowchart.",
    ),
    model_name: str = typer.Option(
        "gpt-4o-mini", help="OpenAI-compatible chat model to use."
    ),
    temperature: float = typer.Option(
        0.2, min=0.0, max=1.0, help="Sampling temperature for the language model."
    ),
    max_iterations: int = typer.Option(
        3,
        min=1,
        help="Maximum self-reflection iterations before returning the latest draft.",
    ),
    output: Path = typer.Option(
        Path("flowchart_output.html"),
        help="Where to save the rendered flowchart HTML.",
        show_default=True,
    ),
    no_render: bool = typer.Option(
        False, "--no-render", help="Disable HTML rendering and only print Mermaid."
    ),
) -> None:
    """Generate a flowchart agent output for the given prompt."""
    config = FlowchartAgentConfig(
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
    )
    agent = FlowchartAgent(config=config)
    result = agent.run(
        prompt,
        save_html=not no_render,
        output_path=str(output),
    )
    typer.echo("Mermaid code:")
    typer.echo("```mermaid")
    typer.echo(result.mermaid_code)
    typer.echo("```")
    if result.html_path:
        typer.echo(f"HTML output saved to: {result.html_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
