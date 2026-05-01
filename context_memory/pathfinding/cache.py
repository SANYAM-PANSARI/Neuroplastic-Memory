"""
Semantic Result Cache — avoids redundant traversals for similar queries.

How it works:
1. After a successful query, the result is cached with its query vector.
2. On new queries, the cache checks if any stored vector is similar enough.
3. If yes, return the cached result instantly (0 LLM calls beyond the initial embed).

Cache lives in memory — resets when the process restarts.
This is intentional: stale cache is worse than no cache.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

from context_memory.types import ContextResult


@dataclass
class CacheEntry:
    query_vector: list[float]
    result: ContextResult
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    """
    In-memory semantic result cache.
    Compares query vectors via cosine similarity.
    Zero external dependencies — pure Python.
    """

    def __init__(self, threshold: float = 0.92, max_size: int = 200, ttl_seconds: float = 3600):
        """
        Args:
            threshold: Cosine similarity threshold for a cache hit (0.92 = very similar).
            max_size: Max entries. Oldest evicted on overflow.
            ttl_seconds: Time-to-live. Entries older than this are evicted.
        """
        self._entries: list[CacheEntry] = []
        self._threshold = threshold
        self._max_size = max_size
        self._ttl = ttl_seconds

    def lookup(self, query_vector: list[float]) -> ContextResult | None:
        """
        Check the cache for a similar query. O(n) scan but n is small (max 200).
        Returns the cached ContextResult or None.
        """
        now = time.time()
        best_entry: CacheEntry | None = None
        best_sim = 0.0

        # Evict expired entries while scanning
        alive: list[CacheEntry] = []
        for entry in self._entries:
            if now - entry.timestamp > self._ttl:
                continue  # Expired — drop it
            alive.append(entry)

            sim = _cosine_similarity(query_vector, entry.query_vector)
            if sim > self._threshold and sim > best_sim:
                best_sim = sim
                best_entry = entry

        self._entries = alive

        if best_entry:
            best_entry.hit_count += 1
            return copy.deepcopy(best_entry.result)  # Return copy to prevent mutation

        return None

    def store(self, query_vector: list[float], result: ContextResult) -> None:
        """Store a successful result in the cache."""
        if result.confidence < 0.3:
            return  # Don't cache low-confidence results

        # Evict oldest if full
        if len(self._entries) >= self._max_size:
            self._entries.sort(key=lambda e: e.timestamp)
            self._entries = self._entries[len(self._entries) // 4:]  # Drop oldest 25%

        self._entries.append(CacheEntry(
            query_vector=query_vector,
            result=copy.deepcopy(result),  # Store copy to prevent caller mutation
            timestamp=time.time(),
        ))

    @property
    def size(self) -> int:
        return len(self._entries)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Pure Python — no numpy needed."""
    if len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))
