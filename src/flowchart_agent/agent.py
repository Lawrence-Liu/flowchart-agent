from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from .models import FlowchartCritique, FlowchartDraft, FlowchartResult
from .prompts import CRITIQUE_PROMPT, DRAFT_PROMPT, REVISION_PROMPT
from .renderer import write_mermaid_html


def _ensure_mermaid_header(mermaid_code: str) -> str:
    stripped = mermaid_code.strip()
    if stripped.lower().startswith("flowchart"):
        return stripped
    return f"flowchart TD\n{stripped}"


@dataclass
class FlowchartAgentConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_iterations: int = 3
    request_timeout: Optional[float] = 60.0


class FlowchartAgent:
    """LangChain-powered agent that drafts, critiques, and revises flowcharts."""

    def __init__(
        self,
        config: FlowchartAgentConfig | None = None,
        llm: Optional[ChatOpenAI] = None,
    ) -> None:
        self.config = config or FlowchartAgentConfig()
        self.llm = llm or ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            timeout=self.config.request_timeout,
        )
        self._draft_parser = PydanticOutputParser(pydantic_object=FlowchartDraft)
        self._critique_parser = PydanticOutputParser(pydantic_object=FlowchartCritique)

        draft_prompt = DRAFT_PROMPT.partial(
            format_instructions=self._draft_parser.get_format_instructions()
        )
        critique_prompt = CRITIQUE_PROMPT.partial(
            format_instructions=self._critique_parser.get_format_instructions()
        )
        revision_prompt = REVISION_PROMPT.partial(
            format_instructions=self._draft_parser.get_format_instructions()
        )

        self._draft_chain = draft_prompt | self.llm | self._draft_parser
        self._critique_chain = critique_prompt | self.llm | self._critique_parser
        self._revision_chain = revision_prompt | self.llm | self._draft_parser

    def run(
        self,
        user_prompt: str,
        *,
        save_html: bool = True,
        output_path: str = "flowchart_output.html",
        config: Optional[RunnableConfig] = None,
    ) -> FlowchartResult:
        """Generate a refined flowchart for the user prompt."""
        draft = self._draft(user_prompt, config=config)
        for iteration in range(self.config.max_iterations):
            critique = self._critique(
                user_prompt, draft.mermaid_code, draft.rationale, config=config
            )
            if critique.is_satisfactory:
                return self._finalize(
                    user_prompt, draft.mermaid_code, save_html, output_path
                )
            draft = self._revise(
                user_prompt,
                draft.mermaid_code,
                critique.revision_guidance,
                config=config,
            )

        # Return the most recent draft if we exit due to iteration limit.
        return self._finalize(
            user_prompt,
            draft.mermaid_code,
            save_html,
            output_path,
        )

    def _draft(
        self, user_prompt: str, *, config: Optional[RunnableConfig]
    ) -> FlowchartDraft:
        try:
            result = self._draft_chain.invoke({"user_prompt": user_prompt}, config=config)
        except ValidationError as exc:
            raise ValueError(f"Draft generation failed: {exc}") from exc
        result.mermaid_code = _ensure_mermaid_header(result.mermaid_code)
        return result

    def _critique(
        self,
        user_prompt: str,
        mermaid_code: str,
        rationale: str,
        *,
        config: Optional[RunnableConfig],
    ) -> FlowchartCritique:
        try:
            return self._critique_chain.invoke(
                {
                    "user_prompt": user_prompt,
                    "mermaid_code": mermaid_code,
                    "rationale": rationale,
                },
                config=config,
            )
        except ValidationError as exc:
            raise ValueError(f"Critique parsing failed: {exc}") from exc

    def _revise(
        self,
        user_prompt: str,
        mermaid_code: str,
        critique: str,
        *,
        config: Optional[RunnableConfig],
    ) -> FlowchartDraft:
        try:
            result = self._revision_chain.invoke(
                {
                    "user_prompt": user_prompt,
                    "mermaid_code": mermaid_code,
                    "critique": critique,
                },
                config=config,
            )
        except ValidationError as exc:
            raise ValueError(f"Revision parsing failed: {exc}") from exc
        result.mermaid_code = _ensure_mermaid_header(result.mermaid_code)
        return result

    def _finalize(
        self, user_prompt: str, mermaid_code: str, save_html: bool, output_path: str
    ) -> FlowchartResult:
        html_path = None
        if save_html:
            html_path = write_mermaid_html(mermaid_code, output_path=output_path)
        return FlowchartResult(
            prompt=user_prompt,
            mermaid_code=mermaid_code.strip(),
            html_path=html_path,
        )
