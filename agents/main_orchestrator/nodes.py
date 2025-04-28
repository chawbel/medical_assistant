import logging
from typing import TypedDict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core.agent_state import AgentState
from core.models import get_llm
from config.settings import settings
from .schemas import RoutingDecision

logger = logging.getLogger(__name__)


def route_logic_by_role(state: AgentState) -> str:
    """Routing logic: determine path based on user role"""
    logger.debug("Executing route_logic_by_role")
    role = state.get("user_role", "unknown")
    if role == "patient":
        return "patient_intent_router"
    elif role == "doctor":
        return "doctor_intent_router"
    else:
        logger.warning(f"unknown user role: {role}. detected in routing logic")
        return "handle_error_node"


def patient_intent_router(state: AgentState) -> AgentState:
    """Uses LLM to determine patient's specific intent"""
    logger.info("Node: patient_intent_router")
    return _determine_intent_llm(
        state,
        settings.PATIENT_INTENT_ROUTER_PROMPT,
        ["supervisor_patient_general", "supervisor_scheduling", "handle_error_node"],
    )


def doctor_intent_router(state: AgentState) -> AgentState:
    """Uses LLM to determine doctor's specific intent"""
    logger.info("Node: doctor_intent_router")
    return _determine_intent_llm(
        state,
        settings.DOCTOR_INTENT_ROUTER_PROMPT,
        ["supervisor_doctor_general", "supervisor_summarization", "handle_error_node"],
    )


def _determine_intent_llm(
    state: AgentState, system_prompt: str, valid_supervisors: list[str]
) -> AgentState:
    """Generic LLM call to determine intent and next supervisor."""
    messages = state.get("messages", [])
    current_query = (
        messages[-1].content
        if messages and isinstance(messages[-1], HumanMessage)
        else ""
    )

    user_role = state.get("user_role")

    if not current_query:
        logger.warning("_determine_intent_llm: No query found.")
        return {
            "supervisor_name": "handle_error_node",
            "route_error": "No query in state.",
        }

    try:
        llm = get_llm("router")
        if not llm:
            raise ValueError("LLM instance not available")
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "User Query: {query}",
                ),  # Role is implicitly handled by which prompt is used
            ]
        )
        parser = JsonOutputParser(pydantic_object=RoutingDecision)
        chain = prompt | llm | parser
    except Exception as e:
        logger.error(
            f"Failed to initialize LLM/Chain for intent routing: {e}", exc_info=True
        )
        return {
            "supervisor_name": "handle_error_node",
            "route_error": "Intent Router init failed.",
        }

    try:
        llm_response = chain.invoke({"query": current_query})
        supervisor = llm_response.get("supervisor")
        reasoning = llm_response.get("reasoning", "N/A")

        if supervisor not in valid_supervisors:
            logger.warning(
                f"LLM returned invalid supervisor for role '{user_role}': '{supervisor}'. Routing to error node."
            )
            return {
                "supervisor_name": "handle_error_node",
                "route_error": f"Invalid routing target for role '{user_role}': {supervisor}",
            }

        logger.info(
            f"Intent Router Decision: Supervisor='{supervisor}', Reasoning='{reasoning}'"
        )
        # Store the decided supervisor name in the state to be used by the next conditional edge
        return {"supervisor_name": supervisor}

    except Exception as e:
        logger.error(f"Error during LLM intent routing or parsing: {e}", exc_info=True)
        return {
            "supervisor_name": "handle_error_node",
            "route_error": f"LLM intent routing failed: {e}",
        }


# --- Placeholder Supervisor Nodes (Functions) ---
# These are called ONLY if the routing points to them. They represent
# supervisor logic that hasn't been built out as a sub-graph yet.
def supervisor_doctor_general_placeholder(state: AgentState) -> AgentState:
    logger.info("Node: supervisor_doctor_general_placeholder")
    return {"final_output": "Placeholder: Doctor general query would be handled here."}


def supervisor_scheduling_placeholder(state: AgentState) -> AgentState:
    logger.info("Node: supervisor_scheduling_placeholder")
    return {"final_output": "Placeholder: Scheduling logic would start here."}


def supervisor_summarization_placeholder(state: AgentState) -> AgentState:
    logger.info("Node: supervisor_summarization_placeholder")
    return {"final_output": "Placeholder: Summarization logic would start here."}


def supervisor_image_analysis_placeholder(state: AgentState) -> AgentState:
    logger.info("Node: supervisor_image_analysis_placeholder")
    # This node would later analyze state['image_path'] and route to specific CV agents
    return {"final_output": "Placeholder: Image analysis logic would start here."}


def handle_error_node(state: AgentState) -> AgentState:
    logger.info("Node: handle_error_node")
    reason = state.get("route_error", "Unknown processing error")
    logger.error(f"Executing error handler. Reason: {reason}")
    return {"final_output": f"Sorry, I encountered an error. Reason: {reason}"}


def check_supervisor_handoff(state: AgentState) -> dict:
    """Checks if a supervisor requested a handoff"""
    logger.info("Node: check_supervisor_handoff")
    next_supervisor = state.get("next_supervisor_required")
    updates = {}
    if next_supervisor:
        logger.info(f"Handoff detected for {next_supervisor} Clearing signlas")
        state["next_supervisor_required"] = None
        updates["next_supervisor_required"] = None
    else:
        logger.info("No handoff detected")
    return updates
