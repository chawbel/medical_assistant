import logging
from typing import Optional

from langchain_cohere import CohereRerank
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from config.settings import settings

logger = logging.getLogger(__name__)

_reranker_compressor_instance: Optional[CohereRerank] = None


def get_reranker() -> Optional[CohereRerank]:
    """
    Initializes and returns a singleton instance of the CohereRerank compressor.
    """
    global _reranker_compressor_instance

    if _reranker_compressor_instance is None:
        logger.info("Initializing Cohere Reranker...")
        if settings.cohere_api_key:
            try:
                _reranker_compressor_instance = CohereRerank(
                    model=settings.reranker,
                    cohere_api_key=settings.cohere_api_key,
                    top_n=settings.rag_reranker_top_k,  # Use top_k from settings (add this to settings.py)
                )
                logger.info(
                    f"Initialized CohereRerank compressor with model '{settings.reranker}'"
                )
            except Exception as e:
                logger.error(f"Failed to initialize CohereRerank: {e}", exc_info=True)
        else:
            logger.warning(
                "COHERE_API_KEY not configured. Reranker cannot be initialized."
            )

    return _reranker_compressor_instance


def get_compression_retriever(
    base_retriever: BaseRetriever,
) -> Optional[ContextualCompressionRetriever]:
    """Creates a ContextualCompressionRetriever using the initialized reranker."""
    reranker = get_reranker()
    if reranker and base_retriever:
        return ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )
    else:
        logger.warning(
            "Cannot create compression retriever: Reranker or base_retriever not available."
        )
        return None


def clear_reranker_instance():
    global _reranker_compressor_instance
    _reranker_compressor_instance = None
