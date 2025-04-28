import logging
from typing import TypedDict

from core.agent_state import AgentState
from config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

logger = logging.getLogger(__name__)


class PatientAnalysisOutput(TypedDict):
    """output schema for the patient analysis LLM call"""

    response_text: str
    request_scheduling: bool


def analyze_patient_query(state: AgentState) -> AgentState:
    """
    Uses an LLM to analyze the patient query, generate a response,
    and decide if scheduling is needed
    """

    logger.info("Node: analyze_patient_query")
    messages = state.get("messages", [])
    # Extracts history excluding the latest user message which is the current query
    history = messages[:-1]
    current_query = (
        messages[-1].content if messages and messages[-1].type == "human" else ""
    )

    if not current_query:
        logger.warning("analyze_patient_query: No user found in the latest message")
        return {
            "patient_response_text": "I'm sorry, I didnt quite catch that. could you please repeat your question?",
            "request_scheduling": False,
        }

    history_str = "\n".join([f"{m.type.upper()}: {m.content}" for m in history])

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17", api_key=settings.google_api_key
        )
        prompt = ChatPromptTemplate.from_template(
            settings.PATIENT_ANALYSIS_SYSTEM_PROMPT
        )
        prompt.input_variables = ["history", "query"]
        parser = JsonOutputParser(pydantic_object=PatientAnalysisOutput)
        chain = prompt | llm | parser

    except Exception as e:
        logger.error(f"Failed to initialize LLM or chaine: {e}", exc_info=True)
        return {
            "patient_response_text": "Sorry, I'm having trouble understanding right now",
            "request_scheuling": False,
        }

    try:
        llm_response: PatientAnalysisOutput = chain.invoke(
            {"history": history_str, "query": current_query}
        )

        response_text = llm_response.get(
            "response_text", "I'm not sure how to respond to that"
        )
        request_scheduling = llm_response.get("reuqest_scheduling", False)

        logger.info(
            f"Patient analysis LLM decision: Scheduling='{request_scheduling}', Response='{response_text[:100]}'"
        )

        return {
            "patient_response_text": response_text,
            "request_scheduling": request_scheduling,
        }

    except Exception as e:
        logger.error(
            f"Error during patient query analysis LLM call: {e}", exc_info=True
        )
        return {
            "patient_response_text": "Sorry I encountered and error while thinking about your request",
            "request_scheduling": False,
        }


def decide_next_step(state: AgentState) -> AgentState:
    """Routes based on whether scheduling was requested"""
    logger.info("Node: decide_next_step")
    if state.get("request_scheduling", False):
        logger.info("Routing to prepare scheduling")
        return "prepare_for_scheduling"
    else:
        logger.info("Routing to finish conversation")
        return "finish_conversation"


def prepare_for_scheduling(state: AgentState) -> AgentState:
    """Sets the state to signal handoff to the scheduling supervisor"""
    logger.info("Node: prepare_for_scheduling")
    response_text = state.get("patient_response_text", "Okay lets look scheduling")
    return {
        "final_output": response_text,
        "next_supervisor_required": "supervisor_scheduling",
    }


def finish_conversation(state: AgentState) -> AgentState:
    """Sets the final output for a normal conversational turn"""
    logger.info("Node: finish_conversation")
    response_text = state.get(
        "patient_response_text", "Sorry Im not sure how to respond"
    )
    return {"final_output": response_text}
