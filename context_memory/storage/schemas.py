"""
Pydantic schemas mirroring all three database stores.
Single source of truth for data shapes — used by storage wrappers and API.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Neo4j: Nodes
# ──────────────────────────────────────────────

class GraphNode(BaseModel):
    """A node in the Neo4j graph. Contains NO text — only structure."""
    node_id: str
    layer: int                                      # 0 = ground truth, 1 = index, 2+ = ephemeral
    node_type: str = "chunk"                        # "chunk", "summary", "cluster"
    dataset_id: str = "default"
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ──────────────────────────────────────────────
# Neo4j: Edges
# ──────────────────────────────────────────────

class GraphEdge(BaseModel):
    """An edge in the Neo4j graph. Carries confidence math and versioning."""
    source_id: str
    target_id: str
    edge_type: str = "parent_child"                 # "parent_child", "semantic_lens", "bridge_lens"
    base_weight: float = 1.0
    context_vector: list[float] = Field(default_factory=list)
    procedural_trace: list[str] = Field(default_factory=list)
    last_traversed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by_agent: str = "default"
    quarantine: bool = True
    version_history: list[dict] = Field(default_factory=list)  # Last 3 states


# ──────────────────────────────────────────────
# Qdrant: Vector Documents
# ──────────────────────────────────────────────

class VectorDocument(BaseModel):
    """A vector entry in Qdrant. Embedding + metadata for scoped search."""
    id: str                                         # Same UUID as the node
    vector: list[float]
    dataset_id: str = "default"
    layer: int = 0


# ──────────────────────────────────────────────
# SQLite/PostgreSQL: Text Payloads
# ──────────────────────────────────────────────

class TextRecord(BaseModel):
    """A row in the relational store. Holds the actual text + provenance."""
    node_id: str
    text_payload: str
    source_uri: str = ""
    drift_score: float = 0.0
    dataset_id: str = "default"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ──────────────────────────────────────────────
# Dataset Configuration
# ──────────────────────────────────────────────

class DatasetConfig(BaseModel):
    """Per-dataset settings, especially decay mode."""
    dataset_id: str
    decay_mode: Literal["drift", "time", "hybrid"] = "drift"
    decay_lambda: float = 0.01
