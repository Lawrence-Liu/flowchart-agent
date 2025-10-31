from __future__ import annotations

import os
import json
from typing_extensions import Annotated, List, NotRequired, TypedDict
from uuid import uuid4
from operator import add

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# --- 0. Environment Setup ---
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# --- 1. State Definition ---

# Define the Agent State, which is the shared memory across the graph
class AgentState(TypedDict):
    """State for the self-refining Mermaid agent."""
    user_input: str  # The original natural language/code input
    mermaid_code: str  # The current generated Mermaid code
    critique_history: Annotated[List[str], add]  # List of past critiques
    revision_number: int  # Current revision count
    reflection_output: NotRequired[Reflection]


# --- 2. Pydantic Schema for Structured Reflection ---

class Reflection(BaseModel):
    """
    Schema for the Reflector to output a structured critique and decision.
    This structure is used to reliably route the conditional edge.
    """
    is_satisfactory: bool = Field(
        description="A boolean indicating if the generated Mermaid code is complete, syntactically valid, and accurate based on the user's requirements."
    )
    critique_or_suggestion: str = Field(
        description="Detailed critique and specific improvement suggestions if not satisfactory. If satisfactory, state 'Looks great! Ready to finalize.' Keep suggestions concise and actionable."
    )


# --- 3. Node Functions ---

def generate_mermaid(state: AgentState) -> AgentState:
    """
    Node that generates or regenerates the Mermaid flowchart code.
    """
    print("\n--- GENERATOR: Starting Generation ---")
    
    # Contextual prompt for regeneration
    history_context = "\n".join(state['critique_history'])
    print("history_context:", history_context)
    if state['revision_number'] > 0:
        system_prompt = (
            "You are an expert Mermaid code generator. The user wants a Mermaid flowchart to represent their request. "
            f"Your previous attempt was critiqued. Use the following critique history to revise the code. "
            "Output ONLY the complete, corrected Mermaid code block, enclosed in triple backticks and the 'mermaid' language tag. "
            "Ensure the output is a `graph TD` or `graph LR` flowchart, using node shapes like [], () etc."
            f"\n\nCRITIQUE HISTORY:\n{history_context}"
        )
    else:
        system_prompt = (
            "You are an expert Mermaid code generator. Your task is to convert the user's request, which may be "
            "natural language, code, or other text, into a clean, syntactically correct Mermaid flowchart. "
            "Output ONLY the complete Mermaid code block, enclosed in triple backticks and the 'mermaid' language tag. "
            "Ensure the output is a `graph TD` or `graph LR` flowchart, using node shapes like [], () etc"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original Request:\n{user_input}\n\nCurrent Mermaid Code (if any):\n{mermaid_code}")
    ])
    
    # Generate the code
    mermaid_chain = prompt | llm
    response = mermaid_chain.invoke({"user_input": state['user_input'], "mermaid_code": state['mermaid_code']})
    
    # Attempt to extract the code block
    try:
        # Simple extraction logic: look for the first and last backtick blocks
        code_block_start = response.content.find("```mermaid")
        code_block_end = response.content.rfind("```")
        
        if code_block_start != -1 and code_block_end != -1 and code_block_end > code_block_start:
            # Extract content between the markers, removing the starting marker
            generated_code = response.content[code_block_start + len("```mermaid"):code_block_end].strip()

        else:
            # Fallback to entire response content if markers are missing
            generated_code = response.content.strip()

        print(f"--- GENERATOR: Generated Code (Revision {state['revision_number'] + 1}) ---\n{generated_code[:200]}...") # Print snippet
        return {
            "mermaid_code": generated_code,
            "revision_number": state['revision_number'] + 1,
            "critique_history": [f"Revision {state['revision_number'] + 1} generated."]
        }
    except Exception as e:
        print(f"Error extracting Mermaid code: {e}")
        # Append error to history and try again if possible
        error_msg = f"Extraction failed. Error: {e}. Attempting to fix in next revision."
        return {
            "mermaid_code": "", # Clear bad code
            "revision_number": state['revision_number'] + 1,
            "critique_history": [error_msg]
        }


def reflect_on_mermaid(state: AgentState) -> AgentState:
    """
    Node that critiques the generated Mermaid code using a structured Pydantic output.
    """
    print(f"\n--- REFLECTOR: Critiquing Revision {state['revision_number']} ---")
    
    # System prompt for the reflector, binding the Pydantic schema
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert Diagram Reflector. Your job is to critique the provided Mermaid flowchart code against the original user request. "
         "Evaluate the syntax, completeness, accuracy, and clarity. You MUST output a JSON object conforming to the Reflection schema."
         ),
        ("human", 
         "Original Request:\n{user_input}\n\n"
         "Mermaid Code to Critique:\n{mermaid_code}"
        )
    ])
    
    # Get structured JSON response
    reflection_chain = reflection_prompt | llm.with_structured_output(Reflection)
    
    try:
        reflection_result = reflection_chain.invoke({"user_input": state['user_input'], "mermaid_code": state['mermaid_code']})
        
        critique = reflection_result.critique_or_suggestion
        print(f"--- REFLECTOR: Decision: {'SATISFACTORY' if reflection_result.is_satisfactory else 'NEEDS REFINEMENT'} ---")
        print(f"Critique: {critique}")

        # Update state with the new critique
        return {
            "critique_history": [critique],
            # Pass the reflection result as a hidden key for conditional routing
            "reflection_output": reflection_result 
        }

    except Exception as e:
        print(f"Error during reflection: {e}")
        # On error, force refinement
        error_critique = f"ERROR: Reflection failed with exception: {e}. Forcing refinement."
        return {
            "critique_history": [error_critique],
            "reflection_output": Reflection(is_satisfactory=False, critique_or_suggestion=error_critique)
        }


# --- 4. Conditional Edge Logic ---

MAX_REVISIONS = 3

def should_continue(state: AgentState) -> str:
    """
    Conditional edge function to determine the next node based on reflection.
    """
    reflection: Reflection = state.get('reflection_output')
    print(reflection)
    # If reflection is satisfactory OR max revisions reached, END
    if reflection and reflection.is_satisfactory:
        return "end"
    
    if state['revision_number'] >= MAX_REVISIONS:
        print(f"\n--- CONDITION: MAX REVISIONS ({MAX_REVISIONS}) REACHED. Ending. ---")
        return "end"
    
    # Otherwise, loop back to the generator for refinement
    print("\n--- CONDITION: Needs Refinement. Looping back to GENERATE. ---")
    return "generate"


# --- 5. Graph Construction and Compilation ---

def create_agent_workflow():
    """
    Builds and compiles the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("generate", generate_mermaid)
    workflow.add_node("reflect", reflect_on_mermaid)
    
    # Set Entry Point
    workflow.set_entry_point("generate")
    
    # Add Edges
    # 1. After generation, always reflect
    workflow.add_edge("generate", "reflect")
    
    # 2. Conditional edge from reflect node
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "generate": "generate",  # Loop back to generator
            "end": END               # Terminate the graph
        }
    )

    # Compile the graph
    app = workflow.compile(checkpointer=MemorySaver())
    print("LangGraph Workflow Compiled successfully.")
    return app

# --- 6. Execution ---

if __name__ == "__main__":
    
    # Create the workflow
    app = create_agent_workflow()
    
    # Example Request (Natural Language)
    prompt =  "Create a simple decision flowchart for finding a movie to watch. Start with [Check Streaming Services]. If [Found Something Good?], go to [Start Watching]. If [Not Found], check [ask a Friend]. If [Friend Suggests One], go back to [Start Watching]. If [Still No Idea], end with [Read a Book Instead]."
    
    initial_state = {
        "user_input": prompt,
        "mermaid_code": "",
        "critique_history": [],
        "revision_number": 0
    }
    
    print("\n" + "="*80)
    print(f"STARTING AGENT FOR REQUEST:\n{prompt}")
    print("="*80)

    # Run the graph
    thread_id = f"flowchart-{uuid4()}"
    final_state = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    print(f"Total Revisions: {final_state['revision_number'] - 1}")
    print(f"Final Critique: {final_state['critique_history'][-1]}")
    
    # Final Output
    final_mermaid_code = final_state['mermaid_code']
    print("\n--- FINAL GENERATED MERMAID CODE ---")
    print(final_mermaid_code)


    # Optional: Displaying the entire history
    print("\n--- REVISION HISTORY ---")
    for i, step in enumerate(final_state['critique_history']):
        print(f"[{i+1}] {step}")

                                                                     
