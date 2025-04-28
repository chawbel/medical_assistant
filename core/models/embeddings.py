import logging
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

_embedding_model_instance: Optional[Embeddings] = None 

def get_embedding_model() -> Optional[Embeddings]:
    """
    Initializes and returns a singleton instance of the embedding model.

    Returns:
        Optional[Embeddings]: configuration from global settings
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        logger.info("Initializing embedding model...")
        try:
            _embedding_model_instance = GoogleGenerativeAIEmbeddings(
                model = settings.rag_embedding_model_name,
                api_key = settings.google_api_key
            )
            logger.info(f"initialized embedding model: {settings.rag_embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
    
    return _embedding_model_instance

def clear_embedding_model_instance():
    global _embedding_model_instance
    _embedding_model_instance = None