from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class FlowchartDraft(BaseModel):
    """Draft flowchart produced by the LLM."""

    mermaid_code: str = Field(..., description="Mermaid flowchart definition.")
    rationale: str = Field(..., description="Reasoning behind the chosen structure.")


class FlowchartCritique(BaseModel):
    """Evaluation of a draft flowchart."""

    is_satisfactory: bool = Field(
        ..., description="True if the flowchart is ready to deliver."
    )
    revision_guidance: str = Field(
        ...,
        description="Concrete feedback to improve the flowchart when it is not ready.",
    )


class FlowchartResult(BaseModel):
    """Final artefact produced for the user."""

    prompt: str
    mermaid_code: str
    html_path: Optional[Path] = None
