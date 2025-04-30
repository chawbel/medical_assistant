import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document

from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import TextLoader

from config.settings import settings
from core.models import get_embedding_model
from agents.rag.vector_store import PGVectorStore
from agents.rag.document_processor import MedicalDocumentProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_extract_text(file_path: Path) -> List[Dict[str, Any]]:
    """Loads a file and extracts text, returning list of content/metadata dicts."""
    extracted_data = []
    file_extension = file_path.suffix.lower()
    source_name = file_path.name

    try:
        if file_extension == ".pdf":
            logger.info(f"Processing PDF: {source_name}")
            # Use unstructured to partition PDF into elements
            elements = partition_pdf(
                filename=str(file_path),
                strategy="hi_res",  # Use "hi_res" for better layout analysis, requires detectron2 potentially
                # strategy="fast", # Use "fast" for quicker processing if layout is simple
                infer_table_structure=True,
                extract_images_in_pdf=False,  # Keep false for now
            )
            # Concatenate text from relevant elements (customize as needed)
            # For now, focus on NarrativeText, Title, ListItem
            # TODO: Handle Tables, Headers, etc. more specifically later
            content = "\n\n".join(
                [
                    str(el)
                    for el in elements
                    if hasattr(el, "category")
                    and el.category in ["NarrativeText", "Title", "ListItem"]
                ]
            )
            if content.strip():
                extracted_data.append(
                    {
                        "content": content,
                        "metadata": {"source": source_name, "file_type": "pdf"},
                    }
                )
            else:
                logger.warning(
                    f"No relevant text elements extracted from PDF: {source_name}"
                )

        elif file_extension == ".txt":
            logger.info(f"Processing TXT: {source_name}")
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()  # Returns a list of LangChain Documents
            if docs:
                # Assuming one doc per .txt file for simplicity
                content = docs[0].page_content
                if content.strip():
                    extracted_data.append(
                        {
                            "content": content,
                            "metadata": {"source": source_name, "file_type": "txt"},
                        }
                    )
                else:
                    logger.warning(f"Empty TXT file: {source_name}")
            else:
                logger.warning(f"Could not load TXT file: {source_name}")

        else:
            logger.warning(f"Skipping unsupported file type: {source_name}")

    except Exception as e:
        logger.error(
            f"Failed to load/extract text from {source_name}: {e}", exc_info=True
        )

    return extracted_data


def main(args):
    # --- Initialization ---
    logger.info("Starting ingestion process...")
    embedder = get_embedding_model()
    if not embedder:
        logger.critical("Embedding model not initialized. Exiting.")
        return

    try:
        vector_store = PGVectorStore(embedding_function=embedder)
    except Exception as e:
        logger.critical(f"Failed to initialize Vector Store: {e}. Exiting.")
        return

    processor = MedicalDocumentProcessor()  # Initialize processor

    # --- Find Files ---
    files_to_process = []
    if args.file:
        file_path = Path(args.file)
        if file_path.is_file():
            files_to_process.append(file_path)
        else:
            logger.error(f"Specified file not found: {args.file}")
            return
    elif args.dir:
        dir_path = Path(args.dir)
        if dir_path.is_dir():
            # Add more extensions if needed
            for ext in ["*.pdf", "*.txt"]:
                files_to_process.extend(list(dir_path.rglob(ext)))
        else:
            logger.error(f"Specified directory not found: {args.dir}")
            return
    else:
        logger.error("No input file or directory specified.")
        return

    if not files_to_process:
        logger.warning("No files found to process.")
        return

    logger.info(f"Found {len(files_to_process)} files to process.")

    # --- Process and Ingest ---
    total_chunks_added = 0
    files_processed_count = 0
    for file_path in files_to_process:
        logger.info(f"--- Processing file: {file_path.name} ---")
        # 1. Load and Extract Text
        extracted_docs = load_and_extract_text(file_path)

        if not extracted_docs:
            logger.warning(f"No content extracted from {file_path.name}. Skipping.")
            continue

        all_chunks_for_file: List[Document] = []
        for doc_data in extracted_docs:
            # 2. Process Content into Chunks
            try:
                chunks = processor.process_document(
                    content=doc_data["content"], initial_metadata=doc_data["metadata"]
                )
                if chunks:
                    all_chunks_for_file.extend(chunks)
            except Exception as e:
                logger.error(
                    f"Failed to process content from {file_path.name}: {e}",
                    exc_info=True,
                )
                continue  # Skip to next file on processing error

        # 3. Add Chunks to Vector Store
        if all_chunks_for_file:
            try:
                vector_store.add_documents(all_chunks_for_file)
                logger.info(
                    f"Added {len(all_chunks_for_file)} chunks from {file_path.name} to vector store."
                )
                total_chunks_added += len(all_chunks_for_file)
                files_processed_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to add chunks from {file_path.name} to vector store: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(f"No processable chunks generated for {file_path.name}.")

    logger.info("--- Ingestion Summary ---")
    logger.info(f"Files processed: {files_processed_count}/{len(files_to_process)}")
    logger.info(f"Total chunks added: {total_chunks_added}")
    logger.info("Ingestion process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into the medical RAG system."
    )
    parser.add_argument("--file", type=str, help="Path to a single file to ingest.")
    parser.add_argument(
        "--dir", type=str, help="Path to a directory containing files to ingest."
    )
    parsed_args = parser.parse_args()

    if not parsed_args.file and not parsed_args.dir:
        parser.error("Either --file or --dir must be specified.")

    main(parsed_args)
