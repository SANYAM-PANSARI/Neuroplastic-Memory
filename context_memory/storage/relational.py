"""
SQLite relational store wrapper.
Holds all text payloads, provenance metadata, and drift scores.
Uses a persistent connection instead of opening/closing per call.
"""

from __future__ import annotations

import os
from datetime import datetime

import aiosqlite

from context_memory.config import settings
from context_memory.storage.schemas import TextRecord


# SQL to create the texts table if it doesn't exist.
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS texts (
    node_id     TEXT PRIMARY KEY,
    text_payload TEXT NOT NULL,
    source_uri  TEXT DEFAULT '',
    drift_score REAL DEFAULT 0.0,
    dataset_id  TEXT DEFAULT 'default',
    created_at  TEXT DEFAULT (datetime('now'))
)
"""


class RelationalStore:
    """Async SQLite wrapper. Uses a persistent connection for performance."""

    def __init__(self) -> None:
        self._db_path = settings.sqlite_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the data directory, table, and open the persistent connection."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(_CREATE_TABLE)
        await self._db.commit()

    async def close(self) -> None:
        """Close the persistent connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def _conn(self) -> aiosqlite.Connection:
        """Get the persistent connection (must call initialize() first)."""
        if self._db is None:
            raise RuntimeError("RelationalStore not initialized. Call initialize() first.")
        return self._db

    async def insert(self, record: TextRecord) -> None:
        """Insert a text record. Ignores duplicates (same node_id)."""
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO texts
                (node_id, text_payload, source_uri, drift_score, dataset_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.node_id,
                record.text_payload,
                record.source_uri,
                record.drift_score,
                record.dataset_id,
                record.created_at.isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_text(self, node_id: str) -> str | None:
        """Fetch the text payload for a given node. Returns None if missing."""
        cursor = await self._conn.execute(
            "SELECT text_payload FROM texts WHERE node_id = ?",
            (node_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_texts_batch(self, node_ids: list[str]) -> dict[str, str]:
        """Fetch text payloads for multiple nodes at once."""
        if not node_ids:
            return {}
        placeholders = ", ".join("?" for _ in node_ids)
        cursor = await self._conn.execute(
            f"SELECT node_id, text_payload FROM texts WHERE node_id IN ({placeholders})",
            node_ids,
        )
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    async def get_source_uri(self, node_id: str) -> str | None:
        """Fetch the provenance URI for a node."""
        cursor = await self._conn.execute(
            "SELECT source_uri FROM texts WHERE node_id = ?",
            (node_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_drift_score(self, node_id: str) -> float:
        """Fetch the drift score for a node. Returns 0.0 if not found."""
        cursor = await self._conn.execute(
            "SELECT drift_score FROM texts WHERE node_id = ?",
            (node_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0.0

    async def update_drift_score(
        self, node_id: str, drift_score: float
    ) -> None:
        """Update the drift score after a source change is detected."""
        await self._conn.execute(
            "UPDATE texts SET drift_score = ? WHERE node_id = ?",
            (drift_score, node_id),
        )
        await self._conn.commit()

    async def update_text_payload(
        self, node_id: str, new_text: str, drift_score: float = 0.0
    ) -> None:
        """Replace text payload and reset drift score."""
        await self._conn.execute(
            "UPDATE texts SET text_payload = ?, drift_score = ? WHERE node_id = ?",
            (new_text, drift_score, node_id),
        )
        await self._conn.commit()

    async def count_records(self) -> int:
        cursor = await self._conn.execute("SELECT count(*) FROM texts")
        row = await cursor.fetchone()
        return row[0] if row else 0
