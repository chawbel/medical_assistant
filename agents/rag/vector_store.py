import logging
import os
from typing import List, Dict, Any, Optional, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import PGVector

from config.settings import settings

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from core.models import get_embedding_model

logger = logging.getLogger(__name__)


class PGVectorStore:
    """
    Manages interactions with the PostgreSQL/PGVector database
    for storing and retrieving document embeddings
    """

    def __init__(self):
        """
        Initializes the PGVectorStore

        Args:
            embedding_function (Embeddings): an initialized LangChain embedding function instance
        """
        self.connection_string = settings.database_url
        self.collection_name = settings.rag_vector_collection_name
        self.embedding_function = get_embedding_model()

        if not self.embedding_function:
            logger.error(
                "Embedding model failed to initialize. Cannot create PGVectorStore."
            )
            raise ValueError("Embedding model not available")

        if not self.connection_string:
            logger.error(
                "DATABASE_URL is not set in settings. Cannot initialize PGVectorStore"
            )
            raise ValueError("Database connection string is required")

        logger.info(f"Initializing PGVectorStore for collection {self.collection_name}")
        try:
            # Initialize the LangChain PGVector wrapper
            # It automatically handles connection and checks/creates the vector extension
            # and table based on the collection name.
            self.store = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
            )
            logger.info("PGVector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialze PGVectorStore {e}", exc_info=True)
            raise

    def add_document(self, documents: Iterable[Document]):
        """
        Adds document to the PGVector store

        Args:
            documents (Iterable[Document]): An iterable of LangChain Document object
                Each document should have `page_content` and `metadata`
        """

        if not documents:
            logger.warning("no documents provided to add")
            return

        try:
            logger.info(
                f"adding {len(documents)} documents to collection '{self.collection_name}'..."
            )
            ids = self.store.add_documents(documents)
            logger.info(f"Successfully added documents with IDs: {ids}")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to PGVector: {e}", exc_info=True)
            raise

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """Performs similarity search and returns documents with scores

        Args:
            query (str): The search query string
            k (int, optional): The number of documents to return
            filter (Optional[Dict[str, Any]], optional): Optional metadata filter dictionary (passed directly to PGVector)

        Returns:
            List[tuple[Document, float]]: A list of tuples, each containing (Document, similarity_score)
            Returns empty list on error
        """
        logger.info(
            f"Performing similarity search for query: '{query[:50]}...', k={k}, filter={filter}"
        )
        try:
            # The store handles embedding the query using the provided embedding_function
            results = self.store.similarity_search_with_score(
                query=query, k=k, filter=filter
            )
            logger.info(f"Found {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return []

    def delete_documents(self, ids: Optional[List[str]] = None) -> bool:
        """
        Deletes documents by their IDs

        Args:
            ids (Optional[List[str]], optional): A list of document IDs to delete.

        Returns:
            True if deletion was successful, False otherwise
        """
        if not ids:
            logger.warning("No document IDs provided for deletion")
            return False
        try:
            logger.warning(
                f"Attempting to delete {len(ids)} documents from collection '{self.collection_name}'"
            )
            self.store.delete(ids=ids)
            logger.info(f"Successfully deleted documents with IDs: {ids}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}", exc_info=True)
            return False

    # def get_collection_count(self) -> int:
    # """Gets the approximate count of items in the collection"""
