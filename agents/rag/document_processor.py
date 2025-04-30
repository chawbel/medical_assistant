import logging
import re
import uuid
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

logger = logging.getLogger(__name__)


class MedicalDocumentProcessor:
    """
    Processes text content into structured Document chunks with metadata.
    """

    def __init__(self):
        """Initializes the processor with settings."""
        self.logger = logging.getLogger(__name__)
        # Get config values directly from imported settings
        self.chunk_size = settings.rag_chunk_size
        self.chunk_overlap = settings.rag_chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # Common separators, starting with the most structured
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,  # Use character length for splitting
            is_separator_regex=False,
        )
        logger.info(
            f"Initialized MedicalDocumentProcessor with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
        # Basic patterns for initial metadata extraction (can be expanded)
        # Example: Simple pattern for detecting document types based on titles/keywords
        self.doc_type_patterns = {
            "guideline": re.compile(
                r"\b(guideline|protocol|recommendation)s?\b", re.IGNORECASE
            ),
            "research_paper": re.compile(
                r"\b(abstract|introduction|methods|results|discussion|conclusion|study|trial|paper)\b",
                re.IGNORECASE,
            ),
            "policy": re.compile(
                r"\b(policy|protocol|manual|procedure)\b", re.IGNORECASE
            ),
            "drug_info": re.compile(
                r"\b(drug|medication|pharmacology|dosage|monograph)\b", re.IGNORECASE
            ),
            "textbook_chapter": re.compile(
                r"\b(chapter|section|overview of|introduction to)\b", re.IGNORECASE
            ),
        }

    def process_document(
        self, content: str, initial_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Processes raw text content into LangChain Document chunks.

        Args:
            content: The full text content of the document.
            initial_metadata: Base metadata (e.g., {'source': filename}).

        Returns:
            A list of LangChain Document objects, each representing a chunk.
        """
        if not content or not isinstance(content, str):
            logger.warning("process_document received empty or invalid content.")
            return []
        if not initial_metadata or "source" not in initial_metadata:
            logger.warning(
                "process_document received insufficient initial_metadata (missing 'source')."
            )
            # Add a default source if missing, though it's better if provided
            initial_metadata = initial_metadata or {}
            initial_metadata.setdefault("source", f"unknown_source_{uuid.uuid4()}")

        logger.info(
            f"Processing document: {initial_metadata.get('source', 'unknown')}, length: {len(content)}"
        )

        # 1. Basic Metadata Extraction from Full Content
        doc_type = self._detect_document_type(content)
        # TODO: Add more extraction (specialty, entities from full doc if needed)

        # 2. Chunk the Text
        try:
            text_chunks = self.text_splitter.split_text(content)
            logger.info(f"Split document into {len(text_chunks)} chunks.")
        except Exception as e:
            logger.error(
                f"Failed to split text for document {initial_metadata['source']}: {e}",
                exc_info=True,
            )
            return []  # Return empty list if chunking fails

        # 3. Create LangChain Document objects for each chunk with metadata
        processed_docs: List[Document] = []
        # Generate a base ID for all chunks from this document
        # Use hash of content + source for more uniqueness if content might repeat across sources
        doc_hash_base = hashlib.md5(
            f"{initial_metadata['source']}_{content}".encode()
        ).hexdigest()

        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():  # Skip empty chunks
                continue

            chunk_metadata = initial_metadata.copy()  # Start with base metadata

            # Add chunk-specific metadata
            # Use parts of the hash and index for a deterministic chunk ID
            chunk_id_str = f"{doc_hash_base[:16]}-{i:04d}"  # Example format
            chunk_metadata["chunk_id"] = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id_str)
            )  # Generate UUID from string
            chunk_metadata["chunk_number"] = i
            chunk_metadata["total_chunks"] = len(text_chunks)
            chunk_metadata["document_type"] = doc_type  # Add detected type
            chunk_metadata["processing_timestamp"] = datetime.now().isoformat()
            # TODO: Add extracted entities specific to *this chunk* later
            # TODO: Add section headers if using section-based chunking

            # Create the LangChain Document
            doc = Document(page_content=chunk_text, metadata=chunk_metadata)
            processed_docs.append(doc)

        logger.info(
            f"Successfully processed document {initial_metadata['source']} into {len(processed_docs)} chunks."
        )
        return processed_docs

    def _detect_document_type(self, text: str) -> str:
        """Detects document type based on simple keyword matching."""
        # Look for keywords in the first ~1000 characters for efficiency
        text_start = text[:1000]
        for doc_type, pattern in self.doc_type_patterns.items():
            if pattern.search(text_start):
                return doc_type
        return "unknown"  # Default if no pattern matches

    # Add other helper methods like _extract_medical_entities_from_chunk later
    # def _extract_medical_entities_from_chunk(self, chunk_text: str) -> Dict[str, List[str]]:
    #     # ... implementation using regex or NER ...
    #     pass
