import logging
from typing import Optional, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings

logger = logging.getLogger(__name__)

# dictionary to hold different llm instances
_llm_instances: Dict[str, Optional[ChatGoogleGenerativeAI]] = {}

DEFAULT_LLM_CONFIG = {
    "router": "gemini-2.0-flash-lite",
    "patient_analyzer": "gemini-2.0-flash",
    "doctor_synthesizer": "gemini-2.5-pro-exp-03-25",
}


def get_llm(instance_name: str = "router") -> Optional[ChatGoogleGenerativeAI]:
    """
    Initializes and returns a specific named LLM instance.
    Uses a shared cache for instances.

    Args:
        instance_name (str, optional): Key to identify the LLM configuration. Defaults to "router".

    Returns:
        Optional[ChatGoogleGenerativeAI]: An initialized ChatGoogleGenerativeAI instance or None if config fails
    """
    global _llm_instances

    if instance_name not in _llm_instances:
        logger.info(f"Initializing LLM instance: '{instance_name}'...")
        try:
            # Get model name from defaults, potentially override with settings later if needed
            model_name = DEFAULT_LLM_CONFIG.get(instance_name, "gemini-2.0-flash")

            instance = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.google_api_key,
                # temperature=temp # Example
                # Add other parameters like generation_config for JSON mode if needed
            )
            _llm_instances[instance_name] = instance
            logger.info(
                f"Initialized LLM instance '{instance_name}' with model '{model_name}'"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize LLM instance '{instance_name}': {e}",
                exc_info=True,
            )
            _llm_instances[instance_name] = None  # Store None on failure

    return _llm_instances.get(
        instance_name
    )  # Return None if initialization failed previously


def clear_llm_instances():
    global _llm_instances
    _llm_instances = {}
