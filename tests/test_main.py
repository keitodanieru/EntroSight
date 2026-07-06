"""Unit/integration tests for the FastAPI application (app/main.py).

These tests avoid loading a real model checkpoint, ChromaDB, or Ollama
service by bypassing the lifespan handler and attaching mock components
directly to ``app.state``. This keeps the tests fast and focused on route
logic, status codes, and response shapes rather than full pipeline behavior
(which is covered separately by scan.py / component tests).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.models import ScanRecord
from app.scan import (
    AppComponents,
    clear_scan_status,
    register_scan,
    update_scan_status,
)


@pytest.fixture
def mock_components() -> AppComponents:
    """Build an AppComponents bundle with mocked pipeline components.

    The FastAPI TestClient used below does not trigger the lifespan handler
    (it is only invoked when used as a context manager), so tests attach
    these mocks to app.state manually instead of relying on real component
    initialization.
    """
    return AppComponents(
        validator=MagicMock(),
        heatmap_generator=MagicMock(),
        classifier=MagicMock(),
        rag_engine=MagicMock(),
        explanation_generator=MagicMock(),
        db=AsyncMock(),
    )


@pytest.fixture
def client(mock_components: AppComponents) -> TestClient:
    """TestClient with mocked settings/components attached, bypassing lifespan."""
    main_module.app.state.settings = MagicMock(
        rag_top_k=5, heatmap_storage_dir="data/heatmaps"
    )
    main_module.app.state.components = mock_components
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _cleanup_scan_status():
    """Ensure test scan IDs don't leak between tests."""
    yield


# ---------------------------------------------------------------------------
# POST /api/scan
# ---------------------------------------------------------------------------


class TestSubmitScan:
    def test_returns_202_with_scan_id(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: uploading a file returns 202 with a scan_id and pending status."""
        # Prevent the real pipeline from running during the test.
        monkeypatch.setattr(main_module, "execute_scan", AsyncMock())

        response = client.post(
            "/api/scan",
            files={"file": ("sample.exe", b"MZ" + b"\x00" * 100, "application/octet-stream")},
        )

        assert response.status_code == 202
        body = response.json()
        assert "scan_id" in body
        assert body["status"] == "pending"

        clear_scan_status(body["scan_id"])

    def test_registers_scan_status(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """After submission, the scan_id should be registered in the status store."""
        from app.scan import get_scan_status

        monkeypatch.setattr(main_module, "execute_scan", AsyncMock())

        response = client.post(
            "/api/scan",
            files={"file": ("sample.exe", b"MZ" + b"\x00" * 100, "application/octet-stream")},
        )
        scan_id = response.json()["scan_id"]

        entry = get_scan_status(scan_id)
        assert entry is not None
        assert entry["scan_id"] == scan_id

        clear_scan_status(scan_id)


# ---------------------------------------------------------------------------
# GET /api/scan/{scan_id}/status
# ---------------------------------------------------------------------------


class TestScanStatus:
    def test_pending_status(self, client: TestClient) -> None:
        register_scan("scan-pending")
        try:
            response = client.get("/api/scan/scan-pending/status")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "pending"
            assert body["result"] is None
        finally:
            clear_scan_status("scan-pending")

    def test_processing_status_with_progress_stage(self, client: TestClient) -> None:
        register_scan("scan-processing")
        update_scan_status("scan-processing", "processing", progress_stage="classifying")
        try:
            response = client.get("/api/scan/scan-processing/status")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "processing"
            assert body["progress_stage"] == "classifying"
        finally:
            clear_scan_status("scan-processing")

    def test_complete_status_includes_result(self, client: TestClient) -> None:
        register_scan("scan-complete")
        record = ScanRecord(
            id=1,
            sha256="a" * 64,
            filename="sample.exe",
            predicted_label="Benign",
            confidence=0.99,
            explanation="Looks benign.",
            heatmap_path="data/heatmaps/a.png",
            timestamp=datetime.utcnow(),
            total_time_ms=123.4,
        )
        update_scan_status("scan-complete", "complete", result=record)
        try:
            response = client.get("/api/scan/scan-complete/status")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "complete"
            assert body["result"]["predicted_label"] == "Benign"
            assert body["result"]["scan_id"] == 1
        finally:
            clear_scan_status("scan-complete")

    def test_error_status_includes_error_message(self, client: TestClient) -> None:
        register_scan("scan-error")
        update_scan_status("scan-error", "error", error_message="Invalid PE file")
        try:
            response = client.get("/api/scan/scan-error/status")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "error"
            assert body["error_message"] == "Invalid PE file"
        finally:
            clear_scan_status("scan-error")

    def test_unknown_scan_id_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/scan/does-not-exist/status")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Template-rendering routes (smoke tests)
# ---------------------------------------------------------------------------


class TestPageRoutes:
    def test_upload_page_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_history_page_returns_200(self, client: TestClient, mock_components: AppComponents) -> None:
        mock_components.db.list_recent = AsyncMock(return_value=[])
        response = client.get("/history")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_scan_result_page_returns_200_when_found(
        self, client: TestClient, mock_components: AppComponents
    ) -> None:
        record = ScanRecord(
            id=1,
            sha256="b" * 64,
            filename="sample.exe",
            predicted_label="Formbook",
            confidence=0.87,
            explanation="Steals credentials.",
            heatmap_path="data/heatmaps/b.png",
            timestamp=datetime.utcnow(),
            total_time_ms=456.7,
        )
        mock_components.db.get_result = AsyncMock(return_value=record)
        response = client.get("/scan/1")
        assert response.status_code == 200

    def test_scan_result_page_returns_404_when_not_found(
        self, client: TestClient, mock_components: AppComponents
    ) -> None:
        mock_components.db.get_result = AsyncMock(return_value=None)
        response = client.get("/scan/999")
        assert response.status_code == 404
