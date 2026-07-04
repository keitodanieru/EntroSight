"""Scan orchestration background task.

Coordinates the full malware-scan pipeline: validate the uploaded file,
generate an entropy heatmap, classify it, retrieve MITRE ATT&CK context,
generate a plain-language explanation, persist the heatmap PNG, and store
the result in the scan history database.

This module is intentionally self-contained: it accepts component instances
as parameters (via ``AppComponents``) rather than importing a global FastAPI
``app`` object, so it can be wired up independently by ``app/main.py`` and
exercised directly in tests.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.components.classifier import MalwareClassifier
from app.components.database import ScanHistoryDB
from app.components.explainer import ExplanationGenerator
from app.components.heatmap import EntropyHeatmapGenerator
from app.components.rag import RAGEngine
from app.components.validator import FileValidator
from app.config import AppSettings
from app.models import ScanRecord

# ---------------------------------------------------------------------------
# Scan status constants
# ---------------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"

STAGE_VALIDATING = "validating"
STAGE_CLASSIFYING = "classifying"
STAGE_EXPLAINING = "explaining"


@dataclass
class AppComponents:
    """Bundle of initialized pipeline components required by ``execute_scan``."""

    validator: FileValidator
    heatmap_generator: EntropyHeatmapGenerator
    classifier: MalwareClassifier
    rag_engine: RAGEngine
    explanation_generator: ExplanationGenerator
    db: ScanHistoryDB


# ---------------------------------------------------------------------------
# In-memory scan status tracking
# ---------------------------------------------------------------------------

# scan_id -> {"scan_id", "status", "progress_stage", "result", "error_message"}
_scan_status_store: dict[str, dict[str, Any]] = {}


def register_scan(scan_id: str) -> None:
    """Register a new scan with ``pending`` status.

    Should be called before the background task is launched so that the
    first status poll never hits a missing scan_id.
    """
    _scan_status_store[scan_id] = {
        "scan_id": scan_id,
        "status": STATUS_PENDING,
        "progress_stage": None,
        "result": None,
        "error_message": None,
    }


def update_scan_status(
    scan_id: str,
    status: str,
    progress_stage: str | None = None,
    result: ScanRecord | None = None,
    error_message: str | None = None,
) -> None:
    """Update the tracked status for a scan.

    Args:
        scan_id: Unique scan identifier.
        status: One of "pending", "processing", "complete", "error".
        progress_stage: One of "validating", "classifying", "explaining", or None.
        result: The completed ScanRecord, if available.
        error_message: Error description, if the scan failed.
    """
    entry = _scan_status_store.setdefault(
        scan_id,
        {
            "scan_id": scan_id,
            "status": STATUS_PENDING,
            "progress_stage": None,
            "result": None,
            "error_message": None,
        },
    )
    entry["status"] = status
    if progress_stage is not None:
        entry["progress_stage"] = progress_stage
    if result is not None:
        entry["result"] = result
    if error_message is not None:
        entry["error_message"] = error_message


def get_scan_status(scan_id: str) -> dict[str, Any] | None:
    """Return the tracked status dict for a scan, or None if unknown."""
    return _scan_status_store.get(scan_id)


def clear_scan_status(scan_id: str) -> None:
    """Remove a scan's tracked status (used in tests / cleanup)."""
    _scan_status_store.pop(scan_id, None)


# ---------------------------------------------------------------------------
# Heatmap persistence helper
# ---------------------------------------------------------------------------


def save_heatmap_png(png_bytes: bytes, file_hash: str, storage_dir: str) -> str:
    """Save heatmap visualization PNG bytes to disk.

    Args:
        png_bytes: PNG image bytes from ``generate_visualization()``.
        file_hash: SHA-256 hash of the scanned file, used as the filename.
        storage_dir: Directory to save the heatmap PNG into.

    Returns:
        The path to the saved PNG file.
    """
    os.makedirs(storage_dir, exist_ok=True)
    heatmap_path = os.path.join(storage_dir, f"{file_hash}.png")
    with open(heatmap_path, "wb") as f:
        f.write(png_bytes)
    return heatmap_path


# ---------------------------------------------------------------------------
# Full scan orchestration
# ---------------------------------------------------------------------------


async def execute_scan(
    scan_id: str,
    file_bytes: bytes,
    filename: str,
    settings: AppSettings,
    components: AppComponents,
) -> ScanRecord | None:
    """Orchestrate the complete scan pipeline as a background task.

    Pipeline: validate -> generate heatmap -> delete file bytes -> classify
    -> retrieve context -> generate explanation -> save heatmap PNG -> store
    in DB.

    Args:
        scan_id: Unique identifier used for status tracking.
        file_bytes: Raw uploaded file content.
        filename: Original filename including extension.
        settings: Application settings.
        components: Initialized pipeline components.

    Returns:
        The stored ScanRecord on success, or None if the scan failed at any
        stage (status is updated to "error" with details in all failure
        cases).
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate
        update_scan_status(scan_id, STATUS_PROCESSING, progress_stage=STAGE_VALIDATING)
        validation = components.validator.validate(filename, file_bytes)
        if not validation.is_valid:
            update_scan_status(
                scan_id, STATUS_ERROR, error_message=validation.error_message
            )
            return None

        # Step 2: Generate heatmap
        update_scan_status(scan_id, STATUS_PROCESSING, progress_stage=STAGE_CLASSIFYING)
        try:
            heatmap_tensor = components.heatmap_generator.generate(file_bytes)
            heatmap_viz_bytes = components.heatmap_generator.generate_visualization(
                file_bytes
            )
        except MemoryError as exc:
            update_scan_status(
                scan_id,
                STATUS_ERROR,
                error_message=f"Memory error during entropy computation: {exc}",
            )
            return None
        finally:
            # Step 3: Delete raw file bytes from memory (privacy-preserving).
            # This drops this function's reference so it can be garbage
            # collected; callers must not retain their own reference either.
            file_bytes = b""

        # Step 4: Classify
        classification = components.classifier.classify(heatmap_tensor)

        # Step 5: Retrieve context (degrade gracefully on RAG failure)
        update_scan_status(scan_id, STATUS_PROCESSING, progress_stage=STAGE_EXPLAINING)
        try:
            passages = components.rag_engine.retrieve_context(
                classification.predicted_label, top_k=settings.rag_top_k
            )
        except Exception:
            passages = []

        # Step 6: Generate explanation
        explanation = await components.explanation_generator.generate(
            label=classification.predicted_label,
            confidence=classification.confidence,
            context_passages=passages,
        )

        # Step 7: Save heatmap visualization
        heatmap_path = save_heatmap_png(
            heatmap_viz_bytes, validation.file_hash, settings.heatmap_storage_dir
        )

        # Step 8: Store in database
        total_time_ms = (time.perf_counter() - start_time) * 1000
        record = ScanRecord(
            id=None,
            sha256=validation.file_hash,
            filename=filename,
            predicted_label=classification.predicted_label,
            confidence=classification.confidence,
            explanation=explanation.explanation_text,
            heatmap_path=heatmap_path,
            timestamp=datetime.utcnow(),
            total_time_ms=total_time_ms,
        )

        try:
            record.id = await components.db.store_result(record)
        except Exception as exc:
            # DB write failure: the scan cannot reach "complete" status
            # without a persisted record (Property 7 - Scan Completeness).
            update_scan_status(
                scan_id,
                STATUS_ERROR,
                error_message=f"Failed to store scan result: {exc}",
            )
            return None

        update_scan_status(scan_id, STATUS_COMPLETE, result=record)
        return record

    except MemoryError as exc:
        update_scan_status(
            scan_id, STATUS_ERROR, error_message=f"Memory error during scan: {exc}"
        )
        return None
    except Exception as exc:  # noqa: BLE001 - top-level safety net for background task
        update_scan_status(scan_id, STATUS_ERROR, error_message=str(exc))
        return None
