"""Unit tests for ScanHistoryDB."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
import pytest_asyncio

from app.components.database import ScanHistoryDB
from app.models import ScanRecord


@pytest_asyncio.fixture
async def db(tmp_path):
    """Create a temporary database for each test."""
    db_path = str(tmp_path / "test_scans.db")
    database = ScanHistoryDB(db_path=db_path)
    await database.initialize()
    yield database
    await database.close()


def _make_record(
    sha256: str = "abc123def456",
    filename: str = "malware.exe",
    predicted_label: str = "AgentTesla",
    confidence: float = 0.95,
    explanation: str = "This is a known RAT.",
    heatmap_path: str = "data/heatmaps/abc123.png",
    total_time_ms: float = 1234.5,
    timestamp: datetime | None = None,
) -> ScanRecord:
    return ScanRecord(
        id=None,
        sha256=sha256,
        filename=filename,
        predicted_label=predicted_label,
        confidence=confidence,
        explanation=explanation,
        heatmap_path=heatmap_path,
        timestamp=timestamp or datetime(2024, 1, 15, 12, 0, 0),
        total_time_ms=total_time_ms,
    )


@pytest.mark.asyncio
async def test_store_and_retrieve(db: ScanHistoryDB):
    """Store a record and retrieve it by ID."""
    record = _make_record()
    record_id = await db.store_result(record)

    assert record_id is not None
    assert isinstance(record_id, int)
    assert record_id > 0

    retrieved = await db.get_result(record_id)
    assert retrieved is not None
    assert retrieved.id == record_id
    assert retrieved.sha256 == record.sha256
    assert retrieved.filename == record.filename
    assert retrieved.predicted_label == record.predicted_label
    assert retrieved.confidence == record.confidence
    assert retrieved.explanation == record.explanation
    assert retrieved.heatmap_path == record.heatmap_path
    assert retrieved.total_time_ms == record.total_time_ms


@pytest.mark.asyncio
async def test_get_result_not_found(db: ScanHistoryDB):
    """get_result returns None for non-existent ID."""
    result = await db.get_result(9999)
    assert result is None


@pytest.mark.asyncio
async def test_get_by_hash(db: ScanHistoryDB):
    """get_by_hash returns the most recent scan for a given hash."""
    record1 = _make_record(
        sha256="deadbeef", timestamp=datetime(2024, 1, 10, 12, 0, 0)
    )
    record2 = _make_record(
        sha256="deadbeef",
        predicted_label="Remcos",
        timestamp=datetime(2024, 1, 15, 12, 0, 0),
    )

    await db.store_result(record1)
    await db.store_result(record2)

    result = await db.get_by_hash("deadbeef")
    assert result is not None
    assert result.sha256 == "deadbeef"
    assert result.predicted_label == "Remcos"  # Most recent


@pytest.mark.asyncio
async def test_get_by_hash_not_found(db: ScanHistoryDB):
    """get_by_hash returns None when hash is not in the database."""
    result = await db.get_by_hash("nonexistent_hash")
    assert result is None


@pytest.mark.asyncio
async def test_list_recent(db: ScanHistoryDB):
    """list_recent returns records ordered by timestamp DESC."""
    for i in range(5):
        record = _make_record(
            sha256=f"hash_{i}",
            filename=f"file_{i}.exe",
            timestamp=datetime(2024, 1, i + 1, 12, 0, 0),
        )
        await db.store_result(record)

    recent = await db.list_recent(limit=3)
    assert len(recent) == 3
    # Most recent first
    assert recent[0].filename == "file_4.exe"
    assert recent[1].filename == "file_3.exe"
    assert recent[2].filename == "file_2.exe"


@pytest.mark.asyncio
async def test_list_recent_empty(db: ScanHistoryDB):
    """list_recent returns empty list when no records exist."""
    recent = await db.list_recent()
    assert recent == []


@pytest.mark.asyncio
async def test_list_recent_default_limit(db: ScanHistoryDB):
    """list_recent defaults to 20 records maximum."""
    for i in range(25):
        record = _make_record(
            sha256=f"hash_{i}",
            timestamp=datetime(2024, 1, 1, i % 24, i % 60, 0),
        )
        await db.store_result(record)

    recent = await db.list_recent()
    assert len(recent) == 20


@pytest.mark.asyncio
async def test_store_returns_unique_ids(db: ScanHistoryDB):
    """Each stored record gets a unique auto-incremented ID."""
    ids = []
    for i in range(3):
        record = _make_record(sha256=f"hash_{i}")
        record_id = await db.store_result(record)
        ids.append(record_id)

    assert len(set(ids)) == 3
    assert ids == sorted(ids)  # Auto-incrementing


@pytest.mark.asyncio
async def test_duplicate_hash_detection(db: ScanHistoryDB):
    """Storing multiple records with the same hash supports deduplication via get_by_hash."""
    same_hash = "duplicate_hash_abc123"

    # Store three records with the same hash at different timestamps
    record1 = _make_record(
        sha256=same_hash,
        filename="sample_v1.exe",
        predicted_label="AgentTesla",
        confidence=0.80,
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
    )
    record2 = _make_record(
        sha256=same_hash,
        filename="sample_v2.exe",
        predicted_label="Remcos",
        confidence=0.85,
        timestamp=datetime(2024, 1, 5, 10, 0, 0),
    )
    record3 = _make_record(
        sha256=same_hash,
        filename="sample_v3.exe",
        predicted_label="DCRat",
        confidence=0.92,
        timestamp=datetime(2024, 1, 10, 10, 0, 0),
    )

    id1 = await db.store_result(record1)
    id2 = await db.store_result(record2)
    id3 = await db.store_result(record3)

    # All three should have distinct IDs (duplicates are stored, not rejected)
    assert len({id1, id2, id3}) == 3

    # get_by_hash returns the most recent record for deduplication
    result = await db.get_by_hash(same_hash)
    assert result is not None
    assert result.sha256 == same_hash
    assert result.predicted_label == "DCRat"  # Most recent by timestamp
    assert result.confidence == 0.92
    assert result.filename == "sample_v3.exe"

    # Each record is still independently retrievable by ID
    r1 = await db.get_result(id1)
    assert r1 is not None
    assert r1.predicted_label == "AgentTesla"


@pytest.mark.asyncio
async def test_schema_creation_is_idempotent(tmp_path):
    """Calling initialize() multiple times does not error."""
    db_path = str(tmp_path / "test.db")
    db = ScanHistoryDB(db_path=db_path)
    await db.initialize()
    await db.initialize()  # Should not raise
    await db.close()
