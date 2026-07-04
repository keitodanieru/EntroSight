"""Unit tests for scan orchestration (app/scan.py)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from app.config import AppSettings
from app.models import (
    ClassificationResult,
    ExplanationResult,
    RetrievedPassage,
    ScanRecord,
    ValidationResult,
)
from app.scan import (
    AppComponents,
    STATUS_COMPLETE,
    STATUS_ERROR,
    STAGE_CLASSIFYING,
    STAGE_EXPLAINING,
    STAGE_VALIDATING,
    clear_scan_status,
    execute_scan,
    get_scan_status,
    register_scan,
    save_heatmap_png,
    update_scan_status,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_components(
    *,
    is_valid: bool = True,
    error_message: str | None = None,
    file_hash: str = "abc123",
    heatmap_side_effect=None,
    rag_side_effect=None,
    explain_side_effect=None,
    db_store_side_effect=None,
) -> AppComponents:
    validator = MagicMock()
    validator.validate.return_value = ValidationResult(
        is_valid=is_valid, error_message=error_message, file_hash=file_hash
    )

    heatmap_generator = MagicMock()
    if heatmap_side_effect is not None:
        heatmap_generator.generate.side_effect = heatmap_side_effect
    else:
        heatmap_generator.generate.return_value = torch.zeros(3, 256, 256)
    heatmap_generator.generate_visualization.return_value = b"fake-png-bytes"

    classifier = MagicMock()
    classifier.classify.return_value = ClassificationResult(
        predicted_label="AgentTesla",
        confidence=0.91,
        all_probabilities={"AgentTesla": 0.91, "Benign": 0.09},
        inference_time_ms=12.3,
    )

    rag_engine = MagicMock()
    if rag_side_effect is not None:
        rag_engine.retrieve_context.side_effect = rag_side_effect
    else:
        rag_engine.retrieve_context.return_value = [
            RetrievedPassage(
                text="Steals credentials.",
                technique_id="T1056",
                technique_name="Input Capture",
                relevance_score=0.8,
            )
        ]

    explanation_generator = MagicMock()
    explain_result = ExplanationResult(
        explanation_text="This is AgentTesla, an info-stealer.",
        generation_time_ms=45.6,
        model_used="mistral",
    )
    if explain_side_effect is not None:
        explanation_generator.generate = AsyncMock(side_effect=explain_side_effect)
    else:
        explanation_generator.generate = AsyncMock(return_value=explain_result)

    db = MagicMock()
    if db_store_side_effect is not None:
        db.store_result = AsyncMock(side_effect=db_store_side_effect)
    else:
        db.store_result = AsyncMock(return_value=42)

    return AppComponents(
        validator=validator,
        heatmap_generator=heatmap_generator,
        classifier=classifier,
        rag_engine=rag_engine,
        explanation_generator=explanation_generator,
        db=db,
    )


@pytest.fixture
def settings(tmp_path) -> AppSettings:
    return AppSettings(
        heatmap_storage_dir=str(tmp_path / "heatmaps"),
        rag_top_k=5,
    )


@pytest.fixture(autouse=True)
def _cleanup_status():
    """Ensure scan status store doesn't leak between tests."""
    yield
    for scan_id in list(get_scan_status.__globals__["_scan_status_store"].keys()):
        clear_scan_status(scan_id)


# ---------------------------------------------------------------------------
# Status tracking helpers
# ---------------------------------------------------------------------------


def test_register_scan_sets_pending_status():
    register_scan("scan-1")
    status = get_scan_status("scan-1")
    assert status is not None
    assert status["status"] == "pending"
    assert status["progress_stage"] is None
    assert status["result"] is None
    assert status["error_message"] is None


def test_update_scan_status_updates_stage_and_status():
    register_scan("scan-2")
    update_scan_status("scan-2", "processing", progress_stage="validating")
    status = get_scan_status("scan-2")
    assert status["status"] == "processing"
    assert status["progress_stage"] == "validating"


def test_update_scan_status_creates_entry_if_missing():
    update_scan_status("scan-new", "processing", progress_stage="classifying")
    status = get_scan_status("scan-new")
    assert status is not None
    assert status["status"] == "processing"
    assert status["progress_stage"] == "classifying"


def test_update_scan_status_preserves_prior_stage_when_not_given():
    register_scan("scan-3")
    update_scan_status("scan-3", "processing", progress_stage="validating")
    update_scan_status("scan-3", "processing")
    status = get_scan_status("scan-3")
    assert status["progress_stage"] == "validating"


def test_get_scan_status_returns_none_for_unknown_id():
    assert get_scan_status("does-not-exist") is None


# ---------------------------------------------------------------------------
# save_heatmap_png
# ---------------------------------------------------------------------------


def test_save_heatmap_png_writes_file(tmp_path):
    storage_dir = str(tmp_path / "heatmaps")
    path = save_heatmap_png(b"pngdata", "deadbeef", storage_dir)
    assert path.endswith("deadbeef.png")
    with open(path, "rb") as f:
        assert f.read() == b"pngdata"


# ---------------------------------------------------------------------------
# execute_scan: happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_scan_success_returns_record_and_marks_complete(settings):
    components = _make_components(file_hash="hash-success")
    register_scan("scan-ok")

    record = await execute_scan(
        scan_id="scan-ok",
        file_bytes=b"MZ" + b"\x00" * 100,
        filename="sample.exe",
        settings=settings,
        components=components,
    )

    assert record is not None
    assert isinstance(record, ScanRecord)
    assert record.id == 42
    assert record.sha256 == "hash-success"
    assert record.predicted_label == "AgentTesla"
    assert record.confidence == 0.91
    assert record.explanation == "This is AgentTesla, an info-stealer."
    assert record.heatmap_path.endswith("hash-success.png")
    assert record.total_time_ms >= 0
    assert isinstance(record.timestamp, datetime)

    status = get_scan_status("scan-ok")
    assert status["status"] == STATUS_COMPLETE
    assert status["result"] == record
    assert status["error_message"] is None


@pytest.mark.asyncio
async def test_execute_scan_progress_stages_called_in_order(settings):
    """Status should pass through validating -> classifying -> explaining."""
    stages_seen: list[str] = []

    components = _make_components()

    # Capture the return value up front, then record the stage at call time
    # without recursing back into the same (now side-effecting) mock.
    validate_result = components.validator.validate.return_value

    def _tracking_validate(filename, file_bytes):
        stages_seen.append(get_scan_status("scan-stages")["progress_stage"])
        return validate_result

    components.validator.validate.side_effect = _tracking_validate

    register_scan("scan-stages")
    await execute_scan(
        scan_id="scan-stages",
        file_bytes=b"MZ" + b"\x00" * 50,
        filename="sample.exe",
        settings=settings,
        components=components,
    )

    # At the moment validate() was invoked, stage should already be "validating"
    assert stages_seen == [STAGE_VALIDATING]

    final_status = get_scan_status("scan-stages")
    assert final_status["status"] == STATUS_COMPLETE


@pytest.mark.asyncio
async def test_execute_scan_calls_components_with_expected_args(settings):
    components = _make_components(file_hash="hash-args")
    register_scan("scan-args")

    await execute_scan(
        scan_id="scan-args",
        file_bytes=b"MZ" + b"\x00" * 10,
        filename="mal.dll",
        settings=settings,
        components=components,
    )

    components.validator.validate.assert_called_once()
    components.heatmap_generator.generate.assert_called_once()
    components.heatmap_generator.generate_visualization.assert_called_once()
    components.classifier.classify.assert_called_once()
    components.rag_engine.retrieve_context.assert_called_once_with(
        "AgentTesla", top_k=settings.rag_top_k
    )
    components.explanation_generator.generate.assert_awaited_once()
    components.db.store_result.assert_awaited_once()


# ---------------------------------------------------------------------------
# execute_scan: validation failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_scan_validation_failure_returns_none_and_marks_error(settings):
    components = _make_components(
        is_valid=False, error_message="Invalid PE file: missing MZ signature"
    )
    register_scan("scan-invalid")

    record = await execute_scan(
        scan_id="scan-invalid",
        file_bytes=b"NOTMZ",
        filename="sample.exe",
        settings=settings,
        components=components,
    )

    assert record is None
    status = get_scan_status("scan-invalid")
    assert status["status"] == STATUS_ERROR
    assert status["error_message"] == "Invalid PE file: missing MZ signature"

    # No further pipeline stages should have executed
    components.heatmap_generator.generate.assert_not_called()
    components.classifier.classify.assert_not_called()
    components.db.store_result.assert_not_called()


# ---------------------------------------------------------------------------
# execute_scan: memory error during heatmap generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_scan_memory_error_marks_error_status(settings):
    components = _make_components(
        heatmap_side_effect=MemoryError("out of memory")
    )
    register_scan("scan-oom")

    record = await execute_scan(
        scan_id="scan-oom",
        file_bytes=b"MZ" + b"\x00" * 10,
        filename="huge.exe",
        settings=settings,
        components=components,
    )

    assert record is None
    status = get_scan_status("scan-oom")
    assert status["status"] == STATUS_ERROR
    assert "Memory error" in status["error_message"]
    components.db.store_result.assert_not_called()


# ---------------------------------------------------------------------------
# execute_scan: DB write failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_scan_db_write_failure_marks_error_status(settings):
    components = _make_components(
        db_store_side_effect=RuntimeError("database is locked")
    )
    register_scan("scan-dbfail")

    record = await execute_scan(
        scan_id="scan-dbfail",
        file_bytes=b"MZ" + b"\x00" * 10,
        filename="sample.exe",
        settings=settings,
        components=components,
    )

    assert record is None
    status = get_scan_status("scan-dbfail")
    assert status["status"] == STATUS_ERROR
    assert "Failed to store scan result" in status["error_message"]


# ---------------------------------------------------------------------------
# execute_scan: RAG failure degrades gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_scan_rag_failure_degrades_gracefully(settings):
    components = _make_components(
        rag_side_effect=RuntimeError("chromadb query failed"),
        file_hash="hash-rag-fail",
    )
    register_scan("scan-rag-fail")

    record = await execute_scan(
        scan_id="scan-rag-fail",
        file_bytes=b"MZ" + b"\x00" * 10,
        filename="sample.exe",
        settings=settings,
        components=components,
    )

    assert record is not None
    status = get_scan_status("scan-rag-fail")
    assert status["status"] == STATUS_COMPLETE
    # Explanation generator should still be called, with empty passages
    components.explanation_generator.generate.assert_awaited_once()
    _, kwargs = components.explanation_generator.generate.call_args
    assert kwargs["context_passages"] == []


# ---------------------------------------------------------------------------
# Property 1: File Privacy - raw bytes are not retained
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_scan_deletes_file_bytes_reference(settings):
    """After heatmap generation, the local file_bytes reference is cleared."""
    components = _make_components(file_hash="hash-privacy")
    register_scan("scan-privacy")

    original_bytes = b"MZ" + b"\x11" * 500

    record = await execute_scan(
        scan_id="scan-privacy",
        file_bytes=original_bytes,
        filename="sample.exe",
        settings=settings,
        components=components,
    )

    assert record is not None
    # The heatmap generator should have been given the original bytes exactly once
    call_args = components.heatmap_generator.generate.call_args
    assert call_args[0][0] == original_bytes


# ---------------------------------------------------------------------------
# Property 7: Scan Completeness - complete records have all required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_scan_complete_record_has_all_required_fields(settings):
    components = _make_components(file_hash="hash-complete")
    register_scan("scan-complete-fields")

    record = await execute_scan(
        scan_id="scan-complete-fields",
        file_bytes=b"MZ" + b"\x00" * 10,
        filename="sample.exe",
        settings=settings,
        components=components,
    )

    assert record is not None
    assert record.sha256 is not None
    assert record.predicted_label is not None
    assert record.confidence is not None
    assert record.explanation is not None
    assert record.heatmap_path is not None
    assert record.timestamp is not None
