"""RAG Engine — ChromaDB retrieval for MITRE ATT&CK technique context."""

from __future__ import annotations

import chromadb
from chromadb.config import Settings

from app.models import Document, RetrievedPassage


class RAGEngine:
    """Retrieves relevant MITRE ATT&CK technique descriptions for malware families."""

    def __init__(
        self,
        collection_name: str = "mitre_attack",
        persist_directory: str = "data/chromadb",
    ):
        """Initialize ChromaDB PersistentClient and load/create collection.

        Args:
            collection_name: Name of the ChromaDB collection to use.
            persist_directory: Path for ChromaDB persistent storage.
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def retrieve_context(
        self, family_label: str, top_k: int = 5
    ) -> list[RetrievedPassage]:
        """Query ChromaDB for relevant MITRE ATT&CK technique descriptions.

        Args:
            family_label: The predicted malware family label (e.g., "AgentTesla").
            top_k: Maximum number of passages to retrieve.

        Returns:
            Ranked list of RetrievedPassage objects with relevance >= 0.3.
        """
        # Handle empty collection gracefully
        if self.collection.count() == 0:
            return []

        # Step 1: Format query
        query_text = f"Malware family {family_label} techniques and behaviors"

        # Step 2: Query ChromaDB with metadata filter
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where={"family": {"$in": [family_label, "general"]}},
        )

        # Handle case where query returns no results
        if (
            not results
            or not results.get("documents")
            or not results["documents"][0]
        ):
            return []

        # Step 3 & 4: Filter by relevance threshold and map to RetrievedPassage
        passages: list[RetrievedPassage] = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns distances; convert to similarity score
            relevance_score = 1.0 - distance
            if relevance_score < 0.3:
                continue
            passages.append(
                RetrievedPassage(
                    text=doc,
                    technique_id=metadata.get("technique_id", "unknown"),
                    technique_name=metadata.get("technique_name", "unknown"),
                    relevance_score=relevance_score,
                )
            )

        return passages

    def ingest_knowledge_base(self, documents: list[Document]) -> int:
        """Ingest MITRE ATT&CK documents into ChromaDB collection.

        Args:
            documents: List of Document objects with text and metadata
                       (technique_id, technique_name, family).

        Returns:
            Number of documents ingested.
        """
        if not documents:
            return 0

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []

        for doc in documents:
            # Use technique_id + family as a stable unique ID for upsert
            technique_id = doc.metadata.get("technique_id", "unknown")
            family = doc.metadata.get("family", "general")
            doc_id = f"{technique_id}_{family}"

            ids.append(doc_id)
            texts.append(doc.text)
            metadatas.append(doc.metadata)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        return len(documents)
