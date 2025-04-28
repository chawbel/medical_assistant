import logging
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Import state and node functions
from core.agent_state import AgentState
from .nodes import (
    route_logic_by_role,
    patient_intent_router,
    doctor_intent_router,
    handle_error_node,
    check_supervisor_handoff,
    supervisor_doctor_general_placeholder,
    supervisor_scheduling_placeholder,
    supervisor_summarization_placeholder,
    supervisor_image_analysis_placeholder,
)

from agents.patient_supervisor import build_patient_supervisor_graph

logger = logging.getLogger(__name__)


def route_after_handoff_check(state: AgentState) -> str:
    """Determines the next node AFTER check_supervisor_handoff node has run."""
    next_supervisor = state.get("next_supervisor_required")
    if next_supervisor:
        # NOTE: check_supervisor_handoff node should ideally clear this signal
        # state["next_supervisor_required"] = None # Modify state ONLY if node logic needs it
        logger.debug(f"Routing after handoff check to: {next_supervisor}")
        return next_supervisor
    else:
        logger.debug("Routing after handoff check to: output_stage")
        return "output_stage"


def build_main_orchestrator_graph():
    """Builds and compiles the main agent orchestrator graph with role based routing."""

    # Build any necessary sub-graphs first
    patient_supervisor_compiled_graph = build_patient_supervisor_graph()

    workflow = StateGraph(AgentState)

    # --- Add Nodes ---
    workflow.add_node("patient_intent_router", patient_intent_router)
    workflow.add_node("doctor_intent_router", doctor_intent_router)

    # Supervisor nodes (Sub-graphs and placeholders)
    workflow.add_node("supervisor_patient_general", patient_supervisor_compiled_graph)
    workflow.add_node(
        "supervisor_doctor_general", supervisor_doctor_general_placeholder
    )
    workflow.add_node("supervisor_scheduling", supervisor_scheduling_placeholder)
    workflow.add_node("supervisor_summarization", supervisor_summarization_placeholder)
    workflow.add_node(
        "supervisor_image_analysis", supervisor_image_analysis_placeholder
    )

    # Utility nodes
    workflow.add_node("handle_error_node", handle_error_node)
    workflow.add_node("check_supervisor_handoff", check_supervisor_handoff)

    # --- Define Graph Structure ---

    # 1. Edges from Role Gate
    workflow.add_conditional_edges(
        START,
        route_logic_by_role,  # function that returns the name of the next node based on state
        {
            "patient_intent_router": "patient_intent_router",
            "doctor_intent_router": "doctor_intent_router",
            "handle_error_node": "handle_error_node",  # Handle unknown roles
        },
    )

    # 2. Conditional Edges from Patient Intent Router
    workflow.add_conditional_edges(
        "patient_intent_router",
        lambda state: state.get("supervisor_name"),  # Route based on LLM decision
        {
            "supervisor_patient_general": "supervisor_patient_general",
            "supervisor_scheduling": "supervisor_scheduling",
            "handle_error_node": "handle_error_node",
            "__default__": "handle_error_node",  # Fallback
        },
    )

    # 3. Conditional Edges from Doctor Intent Router
    workflow.add_conditional_edges(
        "doctor_intent_router",
        lambda state: state.get("supervisor_name"),  # Route based on LLM decision
        {
            "supervisor_doctor_general": "supervisor_doctor_general",
            "supervisor_summarization": "supervisor_summarization",
            "supervisor_image_analysis": "supervisor_image_analysis",
            # Add supervisor_db_agent later
            "handle_error_node": "handle_error_node",
            "__default__": "handle_error_node",  # Fallback
        },
    )

    # 4. Edges AFTER Supervisor Execution to Handoff Check
    # All supervisor nodes must eventually lead here
    workflow.add_edge("supervisor_patient_general", "check_supervisor_handoff")
    workflow.add_edge("supervisor_doctor_general", "check_supervisor_handoff")
    workflow.add_edge("supervisor_scheduling", "check_supervisor_handoff")
    workflow.add_edge("supervisor_summarization", "check_supervisor_handoff")
    workflow.add_edge("supervisor_image_analysis", "check_supervisor_handoff")

    # 5. Conditional Routing AFTER Handoff Check
    workflow.add_conditional_edges(
        "check_supervisor_handoff",
        route_after_handoff_check,  # Use the separate routing logic function
        {  # Mapping: result -> Target Node
            "supervisor_scheduling": "supervisor_scheduling",  # Can handoff back to scheduling
            # Add other handoff targets...
            "output_stage": END,  # Go to END (or guardrails)
            "__default__": "handle_error_node",
        },
    )

    # 6. Final endpoint for explicit errors
    workflow.add_edge("handle_error_node", END)

    # Compile the graph
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    logger.info("Main orchestrator graph compiled.")

    try:
        from PIL import Image
        import io

        png_bytes = graph.get_graph().draw_mermaid_png()

        output_file = "main_orchestrator.png"
        with open(output_file, "wb") as f:
            f.write(png_bytes)
        logger.info(f"Graph visualization saved to {output_file}")
    except ImportError:
        logger.warning("Pillow not installed ")
    except Exception as e:
        logger.error(f"Failed to generate graph visualization: {e}")

    return graph
