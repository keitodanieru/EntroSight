"""Knowledge base loader — ingests MITRE ATT&CK documents into ChromaDB on startup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.components.rag import RAGEngine
from app.models import Document

logger = logging.getLogger(__name__)

DEFAULT_KB_PATH = "data/knowledge_base/mitre_techniques.json"


def load_knowledge_base(
    rag_engine: RAGEngine,
    kb_path: str = DEFAULT_KB_PATH,
) -> int:
    """Load MITRE ATT&CK knowledge base documents into the RAG engine.

    Reads the JSON knowledge base file and ingests all entries as Document
    objects into ChromaDB via the RAGEngine. Intended to be called from
    the FastAPI lifespan handler on startup.

    Args:
        rag_engine: Initialized RAGEngine instance with ChromaDB collection.
        kb_path: Path to the JSON knowledge base file. Defaults to
                 data/knowledge_base/mitre_techniques.json.

    Returns:
        Number of documents ingested into ChromaDB.

    Raises:
        FileNotFoundError: If the knowledge base file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    kb_file = Path(kb_path)
    if not kb_file.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    with kb_file.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    documents: list[Document] = []
    for entry in entries:
        doc = Document(
            text=entry["text"],
            metadata={
                "technique_id": entry["technique_id"],
                "technique_name": entry["technique_name"],
                "family": entry["family"],
            },
        )
        documents.append(doc)

    count = rag_engine.ingest_knowledge_base(documents)
    logger.info("Ingested %d MITRE ATT&CK documents into knowledge base", count)
    return count
