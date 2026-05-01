"""
Context Memory Framework — A neuroplastic AI memory module.

Public API:
    query()   — Retrieve grounded context for a query.
    ingest()  — Ingest new data into the memory graph.
    health()  — System health and statistics.
"""

from context_memory.api import query, ingest, health, close
from context_memory.types import ContextResult, IngestResult, HealthReport

__all__ = [
    "query",
    "ingest",
    "health",
    "close",
    "ContextResult",
    "IngestResult",
    "HealthReport",
]
