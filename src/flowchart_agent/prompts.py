from langchain_core.prompts import ChatPromptTemplate


DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You transform requirements into clean Mermaid flowcharts. "
            "Generate a flowchart that captures the control flow, decisions, and "
            "important outcomes. Focus on clarity and avoid redundant nodes.",
        ),
        (
            "human",
            "Create a flowchart for the following input:\n"
            "```\n{user_prompt}\n```\n"
            "Ensure the chart uses `flowchart TD` or `flowchart LR` and is valid Mermaid.\n"
            "Follow these formatting rules:\n{format_instructions}",
        ),
    ]
)


CRITIQUE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You review Mermaid flowcharts for accuracy and clarity. "
            "Point out concrete defects and decide if the current draft is ready.",
        ),
        (
            "human",
            "Original prompt:\n"
            "```\n{user_prompt}\n```\n"
            "Mermaid draft:\n"
            "```\n{mermaid_code}\n```\n"
            "Rationale:\n"
            "{rationale}\n\n"
            "Respond using:\n{format_instructions}",
        ),
    ]
)


REVISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You revise Mermaid flowcharts using reviewer feedback. Address every issue.",
        ),
        (
            "human",
            "Original prompt:\n"
            "```\n{user_prompt}\n```\n"
            "Previous Mermaid draft:\n"
            "```\n{mermaid_code}\n```\n"
            "Feedback to address:\n"
            "{critique}\n\n"
            "Return output that follows:\n{format_instructions}\n"
            "Preserve valid Mermaid syntax.",
        ),
    ]
)
