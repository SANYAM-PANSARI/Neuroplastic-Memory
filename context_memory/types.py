"""
All public types used by the module API and internal components.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Priority(IntEnum):
    """Query priority levels. Lower value = higher priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class ContextResult:
    """Returned by query(). Everything the caller needs."""
    chunks: list[str]                           # Ranked, token-budgeted payloads
    journey_summary: str                        # Compressed reasoning trace
    confidence: float                           # Aggregate score (0.0–1.0)
    sources: list[str]                          # Provenance URIs
    traversal_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    """Returned by ingest(). Summary of what was ingested."""
    chunks_created: int
    nodes_created: int
    glossary_terms_added: int
    duration_ms: float


@dataclass
class HealthReport:
    """Returned by health(). System introspection."""
    total_nodes: int
    total_edges: int
    total_datasets: int
    active_traversals: int
    write_queue_depth: int
    avg_query_latency_ms: float
    glossary_last_updated: str
    janitor_last_run: str
