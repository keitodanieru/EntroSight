"""Scan history database — async SQLite storage for scan results."""

from __future__ import annotations

import os
from datetime import datetime

import aiosqlite

from app.models import ScanRecord

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scan_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    explanation TEXT NOT NULL,
    heatmap_path TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_time_ms REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scan_results_sha256 ON scan_results(sha256);
CREATE INDEX IF NOT EXISTS idx_scan_results_timestamp ON scan_results(timestamp DESC);
"""


class ScanHistoryDB:
    """Async SQLite database for persisting and querying scan results."""

    def __init__(self, db_path: str = "data/scans.db") -> None:
        """Initialize with the database file path.

        Call ``await initialize()`` before using any other methods.
        """
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database connection and ensure the schema exists."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store_result(self, result: ScanRecord) -> int:
        """Store a completed scan result. Returns the generated record ID."""
        assert self._db is not None, "Database not initialized. Call initialize() first."

        cursor = await self._db.execute(
            """
            INSERT INTO scan_results
                (sha256, filename, predicted_label, confidence,
                 explanation, heatmap_path, timestamp, total_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.sha256,
                result.filename,
                result.predicted_label,
                result.confidence,
                result.explanation,
                result.heatmap_path,
                result.timestamp.isoformat(),
                result.total_time_ms,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_result(self, scan_id: int) -> ScanRecord | None:
        """Retrieve a scan result by its ID."""
        assert self._db is not None, "Database not initialized. Call initialize() first."

        cursor = await self._db.execute(
            "SELECT * FROM scan_results WHERE id = ?", (scan_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    async def get_by_hash(self, sha256: str) -> ScanRecord | None:
        """Look up the most recent scan result by SHA-256 hash (deduplication)."""
        assert self._db is not None, "Database not initialized. Call initialize() first."

        cursor = await self._db.execute(
            "SELECT * FROM scan_results WHERE sha256 = ? ORDER BY timestamp DESC LIMIT 1",
            (sha256,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    async def list_recent(self, limit: int = 20) -> list[ScanRecord]:
        """Return the most recent *limit* scan results ordered by timestamp DESC."""
        assert self._db is not None, "Database not initialized. Call initialize() first."

        cursor = await self._db.execute(
            "SELECT * FROM scan_results ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def search(self, query: str, limit: int = 20) -> list[ScanRecord]:
        """Search scan history by filename, SHA-256 hash, or predicted label.

        Performs a case-insensitive partial match against the filename,
        sha256, and predicted_label columns.

        Args:
            query: Search term to match against scan records.
            limit: Maximum number of results to return.

        Returns:
            List of matching ScanRecord objects ordered by timestamp DESC.
        """
        assert self._db is not None, "Database not initialized. Call initialize() first."

        like_pattern = f"%{query}%"
        cursor = await self._db.execute(
            """
            SELECT * FROM scan_results
            WHERE filename LIKE ? OR sha256 LIKE ? OR predicted_label LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (like_pattern, like_pattern, like_pattern, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> ScanRecord:
        """Convert a database row to a ScanRecord dataclass."""
        return ScanRecord(
            id=row["id"],
            sha256=row["sha256"],
            filename=row["filename"],
            predicted_label=row["predicted_label"],
            confidence=row["confidence"],
            explanation=row["explanation"],
            heatmap_path=row["heatmap_path"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            total_time_ms=row["total_time_ms"],
        )
