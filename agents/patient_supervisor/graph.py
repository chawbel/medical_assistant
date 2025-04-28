import logging
from langgraph.graph import StateGraph, END

from .nodes import (
    analyze_patient_query,
    decide_next_step,
    prepare_for_scheduling,
    finish_conversation,
)
from core.agent_state import AgentState 

logger = logging.getLogger(__name__)


def build_patient_supervisor_graph():
    """Builds and compiles the patient general assistant supervisor graph."""
    workflow = StateGraph(AgentState)

    # Add nodes using the imported functions
    workflow.add_node("analyze_patient_query", analyze_patient_query)
    workflow.add_node("prepare_for_scheduling", prepare_for_scheduling)
    workflow.add_node("finish_conversation", finish_conversation)

    # Set entry point
    workflow.set_entry_point("analyze_patient_query")

    # Define edges
    #workflow.add_edge("analyze_patient_query", "decide_next_step")

    workflow.add_conditional_edges(
        "analyze_patient_query",
        decide_next_step,  # Pass state to the function
        {
            "prepare_for_scheduling": "prepare_for_scheduling",
            "finish_conversation": "finish_conversation",
        },
    )

    workflow.add_edge("prepare_for_scheduling", END)
    workflow.add_edge("finish_conversation", END)

    graph = workflow.compile()
    logger.info("Patient Supervisor graph compiled.")
    return graph
