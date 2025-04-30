import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# define the base dir for the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    google_api_key: str = os.getenv("GOOGLE_API_KEY")

    database_url: str = os.getenv("DATABASE_URL")

    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")

    cohere_api_key: str = os.getenv("COHERE_API_KEY")

    rag_embedding_model_name: str = "models/text-embedding-004"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_embedding_dim: int = 768
    rag_vector_collection_name: str = "medical_documents"
    rag_confidence_threshold: float = 0.7
    rag_reranker_top_k: int = 3
    rag_chunking_strategy: str = "recursive"

    reranker: str = "rerank-v3.5"

    cv_chest_xray_model_path: str = str(
        BASE_DIR / "agents" / "image_analysis" / "models" / "chest_xray_model.pth"
    )
    cv_skin_lesion_model_path: str = str(
        BASE_DIR / "agents" / "image_analysis" / "models" / "skin_lesion_model.pth"
    )

    PATIENT_INTENT_ROUTER_PROMPT: str = """You are a dispatcher for a PATIENT using a medical assistant.
Analyze the patient's query and decide which task they are trying to accomplish.

Available Tasks for Patients:
1.  `supervisor_patient_general`: General conversation, asking about symptoms, non-urgent medical questions, follow-ups. Use this as the default.
2.  `supervisor_scheduling`: Explicit requests to schedule, check, or modify an appointment.
3.  `handle_error_node`: If the query is completely unclear or nonsensical.

Decision Factors:
- Focus on keywords like "appointment", "schedule", "book", "when are you free" for scheduling.
- Otherwise, assume general conversation or inquiry.
- If the query mentions symptoms or asks medical questions, route to `supervisor_patient_general`.

Output Format:
You MUST respond ONLY with a valid JSON object matching this structure:
{{
    "supervisor": "NAME_OF_THE_CHOSEN_SUPERVISOR_NODE",
    "reasoning": "Brief explanation for choosing this task."
}}
"""
    DOCTOR_INTENT_ROUTER_PROMPT: str = """You are a dispatcher for a DOCTOR using a medical assistant AI.
Analyze the doctor's query and decide which task they are trying to accomplish.

Available Tasks for Doctors:
1.  `supervisor_doctor_general`: General conversation, asking medical questions (will trigger RAG/Web search inside), research lookups. Use this as the default.
2.  `supervisor_scheduling`: Requests to check or manage their own schedule or potentially book patient appointments.
3.  `supervisor_summarization`: Explicit requests to summarize text, notes, or patient records.
4.  `supervisor_db_agent`: Requests to fetch specific patient data or records (e.g., "show me patient X's labs", "list patients with diabetes"). <-- We will implement this supervisor later
5.  `supervisor_image_analysis`: Requests involving uploaded images. <-- We will implement this supervisor later
6.  `handle_error_node`: If the query is completely unclear or nonsensical for a doctor's context.

Decision Factors:
- Prioritize explicit commands: "summarize", "schedule", "show record", "analyze image".
- Keywords like "RAG", "search", "latest study", "treatment for", "what is" suggest `supervisor_doctor_general` (which handles RAG/Web).
- General questions or conversation default to `supervisor_doctor_general`.

Output Format:
You MUST respond ONLY with a valid JSON object matching this structure:
{{
    "supervisor": "NAME_OF_THE_CHOSEN_SUPERVISOR_NODE",
    "reasoning": "Brief explanation for choosing this task."
}}
"""

    PATIENT_ANALYSIS_SYSTEM_PROMPT: str = """You are an empathetic and helpful AI medical assistant conversing with a PATIENT.
    Your primary goals are:
    1.  Understand the patient's query or statement.
    2.  Respond conversationally and empathetically.
    3.  If the patient describes symptoms, ask relevant clarifying questions (but do not diagnose).
    4.  Based on the conversation (especially symptoms mentioned or direct requests), determine if scheduling an appointment is the appropriate next step.
    5.  If scheduling IS appropriate OR the patient explicitly asks to schedule, set 'request_scheduling' to true and provide a brief transition message (e.g., "Okay, I can help you with scheduling.").
    6.  If scheduling is NOT the next step, provide a helpful conversational response and set 'request_scheduling' to false.

    DO NOT:
    -   Provide medical diagnoses.
    -   Prescribe medication.
    -   Give definitive medical advice. Always recommend consulting a doctor for such matters.

    Conversation History (if any):
    {history}

    Current User Query:
    {query}

    Output Format:
    You MUST respond ONLY with a valid JSON object matching this structure:
    {{
        "response_text": "Your conversational reply to the patient (or a transition message if scheduling).",
        "request_scheduling": boolean (true if scheduling is the next step, false otherwise)
    }}
    """


settings = Settings()
