"""
Public API surface for the Context Memory Framework.
Exposes query(), ingest(), and health().
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from context_memory.ingestion.pipeline import ingest_file
from context_memory.pathfinding.traversal import find_context
from context_memory.storage.graph import GraphStore
from context_memory.storage.relational import RelationalStore
from context_memory.storage.vector import VectorStore
from context_memory.types import ContextResult, HealthReport, IngestResult


# ── Shared storage instances ──────────────────
_graph: GraphStore | None = None
_vector: VectorStore | None = None
_relational: RelationalStore | None = None
_init_lock = asyncio.Lock()


async def _get_stores() -> tuple[GraphStore, VectorStore, RelationalStore]:
    """Lazy-init and return the 3 storage wrappers. Thread/task-safe."""
    global _graph, _vector, _relational
    if _graph is not None:
        return _graph, _vector, _relational
    async with _init_lock:
        if _graph is None:  # Double-check after acquiring lock
            _graph = GraphStore()
            _vector = VectorStore()
            _relational = RelationalStore()
            await _relational.initialize()
            await _vector.ensure_collection()
            await _vector.ensure_payload_index()
            await _graph.ensure_indexes()
    return _graph, _vector, _relational


async def query(
    query_text: str,
    dataset_id: str,
    agent_id: str = "default",
    max_tokens: int = 4000,
) -> ContextResult:
    """
    Retrieve grounded context for a query.
    Runs the full Active Pathfinding pipeline.
    """
    graph, vector, relational = await _get_stores()
    return await find_context(
        query=query_text,
        dataset_id=dataset_id,
        agent_id=agent_id,
        max_tokens=max_tokens,
        graph=graph,
        vector=vector,
        relational=relational,
    )


async def ingest(
    source_uri: str,
    dataset_id: str,
    embed_delay: float = 2.0,
) -> IngestResult:
    """
    Ingest new data into the memory graph.

    Args:
        source_uri: Path to a file or directory.
        dataset_id: Which dataset to ingest into.
        embed_delay: Seconds between embedding calls (for rate limits).
    """
    graph, vector, relational = await _get_stores()
    path = Path(source_uri)

    if path.is_file():
        return await ingest_file(
            file_path=str(path),
            dataset_id=dataset_id,
            graph=graph,
            vector=vector,
            relational=relational,
            embed_delay=embed_delay,
        )
    elif path.is_dir():
        # Ingest all markdown/text files in directory
        total = IngestResult(
            chunks_created=0, nodes_created=0,
            glossary_terms_added=0, duration_ms=0.0,
        )
        for ext in ("*.md", "*.txt", "*.rst"):
            for file in sorted(path.glob(f"**/{ext}")):
                result = await ingest_file(
                    file_path=str(file),
                    dataset_id=dataset_id,
                    graph=graph,
                    vector=vector,
                    relational=relational,
                    embed_delay=embed_delay,
                )
                total.chunks_created += result.chunks_created
                total.nodes_created += result.nodes_created
                total.duration_ms += result.duration_ms
                total.glossary_terms_added += result.glossary_terms_added
                print(f"  ✓ {file.name}: {result.chunks_created} chunks")
        return total
    else:
        raise FileNotFoundError(f"Source not found: {source_uri}")


async def ingest_topology(
    nodes: list[dict],
    edges: list[dict],
    edge_type: str = "IMPLIES",
    batch_size: int = 5000,
) -> IngestResult:
    """
    Directly ingest a topological mathematical graph (Bring-Your-Own-Data).
    Bypasses markdown chunking and embedding, inserting structure directly.
    """
    graph, _, _ = await _get_stores()
    
    start = asyncio.get_event_loop().time()
    
    # 1. Insert Nodes
    if nodes:
        await graph.bulk_create_nodes(nodes, batch_size=batch_size)
    
    # 2. Insert Edges
    if edges:
        await graph.bulk_create_topology(edges, rel_type=edge_type, batch_size=batch_size)
        
    duration = (asyncio.get_event_loop().time() - start) * 1000
    
    return IngestResult(
        chunks_created=0,
        nodes_created=len(nodes),
        glossary_terms_added=0,
        duration_ms=duration,
        error_count=0
    )


async def health() -> HealthReport:
    """System health and statistics."""
    graph, vector, relational = await _get_stores()
    return HealthReport(
        total_nodes=await graph.count_nodes(),
        total_edges=await graph.count_edges(),
        total_datasets=await graph.count_datasets(),
        active_traversals=0,
        write_queue_depth=0,
        avg_query_latency_ms=0.0,
        glossary_last_updated="never",
        janitor_last_run="never",
    )


async def close() -> None:
    """Clean shutdown of all database connections."""
    global _graph, _vector, _relational
    if _graph:
        await _graph.close()
    if _vector:
        await _vector.close()
    if _relational:
        await _relational.close()
    _graph = _vector = _relational = None
