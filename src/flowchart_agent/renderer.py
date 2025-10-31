from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


_ENV = Environment(
    loader=FileSystemLoader(searchpath=str(Path(__file__).parent)),
    autoescape=select_autoescape(["html"]),
)


def write_mermaid_html(mermaid_code: str, output_path: Path | str) -> Path:
    """Render Mermaid code inside a standalone HTML page."""
    target = Path(output_path)
    template = _ENV.from_string(
        """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Flowchart</title>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({ startOnLoad: true, securityLevel: "loose" });
    </script>
    <style>
      body { font-family: system-ui, sans-serif; background: #f8f9fb; margin: 0; padding: 2rem; }
      .container { background: #fff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08); }
      pre { background: #1e1f22; color: #f5f5f5; padding: 1rem; border-radius: 8px; overflow-x: auto; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Flowchart Diagram</h1>
      <div class="mermaid">
{{ mermaid_code }}
      </div>
      <h2>Source</h2>
      <pre><code>{{ mermaid_code }}</code></pre>
    </div>
  </body>
</html>
"""
    )
    rendered = template.render(mermaid_code=mermaid_code)
    target.write_text(rendered, encoding="utf-8")
    return target
