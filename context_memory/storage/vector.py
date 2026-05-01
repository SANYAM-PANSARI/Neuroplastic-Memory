"""
Qdrant vector database wrapper.
Handles embedding storage and similarity search, scoped by dataset_id.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from context_memory.config import settings


# Default collection name — one collection for all datasets,
# scoped via payload filter on dataset_id.
COLLECTION_NAME = "context_memory"


class VectorStore:
    """Async Qdrant wrapper. All vector reads and writes go through here."""

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(url=settings.qdrant_url)

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection(self, vector_size: int = 3072) -> None:
        """Create the collection if it doesn't exist."""
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if COLLECTION_NAME not in names:
            await self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def ensure_payload_index(self) -> None:
        """Create payload index on dataset_id for fast filtered searches."""
        try:
            await self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="dataset_id",
                field_schema="keyword",
            )
        except Exception:
            pass  # Index may already exist

    async def upsert(
        self,
        point_id: str,
        vector: list[float],
        dataset_id: str,
        layer: int = 0,
    ) -> None:
        """Insert or update a vector with dataset_id metadata."""
        await self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "dataset_id": dataset_id,
                        "layer": layer,
                    },
                )
            ],
        )

    async def search(
        self,
        query_vector: list[float],
        dataset_id: str,
        top_k: int = 3,
        layer_filter: int | None = None,
    ) -> list[dict]:
        """
        Semantic similarity search, scoped to a dataset_id.
        Returns [{id, score, payload}, ...] sorted by similarity.
        """
        must_conditions = [
            FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id))
        ]
        if layer_filter is not None:
            must_conditions.append(
                FieldCondition(key="layer", match=MatchValue(value=layer_filter))
            )

        results = await self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(must=must_conditions),
            limit=top_k,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in results.points
        ]

    # ── Shortcut Vector Operations (Layer 2) ──────

    async def upsert_shortcut(
        self,
        shortcut_id: str,
        vector: list[float],
        dataset_id: str,
        source_id: str,
        target_id: str,
    ) -> None:
        """
        Store a shortcut vector in Qdrant (layer=2).
        Uses HNSW index for O(log N) similarity search instead of
        linear scan on Neo4j edge properties.
        """
        await self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=shortcut_id,
                    vector=vector,
                    payload={
                        "dataset_id": dataset_id,
                        "layer": 2,
                        "source_id": source_id,
                        "target_id": target_id,
                    },
                )
            ],
        )

    async def search_shortcuts(
        self,
        query_vector: list[float],
        dataset_id: str,
        threshold: float = 0.75,
    ) -> dict | None:
        """
        Find the best matching shortcut via HNSW similarity search.
        O(log N) instead of O(N) — scales to millions of shortcuts.
        Returns {target_id, source_id, similarity} or None.
        """
        results = await self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(must=[
                FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id)),
                FieldCondition(key="layer", match=MatchValue(value=2)),
            ]),
            limit=1,
        )

        if results.points and results.points[0].score >= threshold:
            point = results.points[0]
            return {
                "target_id": point.payload["target_id"],
                "source_id": point.payload["source_id"],
                "similarity": point.score,
            }
        return None
