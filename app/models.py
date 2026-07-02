"""Pydantic request/response models and dataclasses for EntroSight."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic API response models
# ---------------------------------------------------------------------------


class ScanResponse(BaseModel):
    """Complete scan result returned to the UI."""

    scan_id: int
    sha256: str
    filename: str
    predicted_label: str
    confidence: float
    explanation: str
    heatmap_url: str
    timestamp: datetime
    total_time_ms: float


class ScanStatusResponse(BaseModel):
    """Polling response for async scan status."""

    scan_id: str
    status: str  # "pending", "processing", "complete", "error"
    progress_stage: str | None = None  # "validating", "classifying", "explaining"
    result: ScanResponse | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Internal dataclass models (component interfaces)
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of PE file validation."""

    is_valid: bool
    error_message: str | None = None
    file_hash: str | None = None  # SHA-256


@dataclass
class ClassificationResult:
    """Result of ResNet50 malware classification."""

    predicted_label: str
    confidence: float  # 0.0 to 1.0
    all_probabilities: dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0


@dataclass
class RetrievedPassage:
    """A single passage retrieved from the MITRE ATT&CK knowledge base."""

    text: str
    technique_id: str  # e.g., "T1055"
    technique_name: str
    relevance_score: float


@dataclass
class ExplanationResult:
    """Result of LLM-generated explanation."""

    explanation_text: str
    generation_time_ms: float
    model_used: str


@dataclass
class ScanRecord:
    """A persisted scan result stored in the database."""

    id: int | None
    sha256: str
    filename: str
    predicted_label: str
    confidence: float
    explanation: str
    heatmap_path: str  # Path to saved heatmap PNG
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_time_ms: float = 0.0


@dataclass
class Document:
    """A document for RAG knowledge base ingestion."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)  # technique_id, technique_name, family
