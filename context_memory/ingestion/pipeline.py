"""
Ingestion pipeline — orchestrates the full ingest flow.

COST BREAKDOWN PER FILE:
  - N embed calls (1 per chunk) — unavoidable
  - 1 summarize call (only if >3 chunks, otherwise uses first chunk as summary)
  - 1 embed call for the summary
  Total: N+1 embeds + 0-1 LLM calls

OPTIMIZATION NOTES:
  - Deduplication: skips chunks already in SQLite (content-based UUIDv5)
  - Small files (<4 chunks): first chunk IS the summary, saving 1 LLM call
  - Average embed calls per chunk for the summary: amortized across all chunks
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from context_memory.agents.llm import embed_batch, summarize, count_tokens
from context_memory.config import settings
from context_memory.ingestion.chunker import Chunk, chunk_markdown
from context_memory.storage.graph import GraphStore
from context_memory.storage.relational import RelationalStore
from context_memory.storage.schemas import GraphNode, TextRecord
from context_memory.storage.vector import VectorStore
from context_memory.types import IngestResult


async def ingest_file(
    file_path: str,
    dataset_id: str,
    graph: GraphStore,
    vector: VectorStore,
    relational: RelationalStore,
    embed_delay: float = 2.0,
) -> IngestResult:
    """
    Ingest a single file. Cost-optimized:
    - Skips already-ingested chunks (dedup by content hash)
    - Skips LLM summary for small files (uses first chunk)
    - Batches embeds with configurable delay for rate limits
    """
    start = time.time()
    path = Path(file_path)

    # 1. Read and chunk (0 API calls)
    text = path.read_text(encoding="utf-8")
    source_uri = str(path.resolve())
    chunks = chunk_markdown(text, source_uri=source_uri)

    if not chunks:
        return IngestResult(
            chunks_created=0, nodes_created=0,
            glossary_terms_added=0, duration_ms=0.0,
        )

    # 2. Deduplicate — skip chunks already in the database
    new_chunks: list[Chunk] = []
    for chunk in chunks:
        node_id = _make_node_id(chunk.text)
        existing = await relational.get_text(node_id)
        if existing is None:
            new_chunks.append(chunk)

    if not new_chunks:
        # All chunks already ingested — skip entirely
        duration_ms = (time.time() - start) * 1000
        return IngestResult(
            chunks_created=0, nodes_created=0,
            glossary_terms_added=0, duration_ms=round(duration_ms, 1),
        )

    # 3. Generate embeddings ONLY for new chunks
    chunk_texts = [c.text for c in new_chunks]
    embeddings = await embed_batch(chunk_texts, delay=embed_delay)

    # 4. Write each new chunk to all 3 stores (Layer 0)
    node_ids: list[str] = []
    for chunk, embedding in zip(new_chunks, embeddings):
        node_id = _make_node_id(chunk.text)
        node_ids.append(node_id)

        await graph.create_node(GraphNode(
            node_id=node_id, layer=0,
            node_type="chunk", dataset_id=dataset_id,
        ))
        await vector.upsert(
            point_id=node_id, vector=embedding,
            dataset_id=dataset_id, layer=0,
        )
        await relational.insert(TextRecord(
            node_id=node_id, text_payload=chunk.text,
            source_uri=chunk.source_uri, dataset_id=dataset_id,
        ))

    # 5. Layer 1 summary — cost-optimized
    summary_text, summary_cost = await _make_summary(chunk_texts)
    summary_id = _make_node_id(summary_text)

    # Check if this summary already exists
    existing_summary = await relational.get_text(summary_id)
    if existing_summary is None:
        await graph.create_node(GraphNode(
            node_id=summary_id, layer=1,
            node_type="summary", dataset_id=dataset_id,
        ))

        summary_embedding = (await embed_batch([summary_text], delay=0))[0]
        await vector.upsert(
            point_id=summary_id, vector=summary_embedding,
            dataset_id=dataset_id, layer=1,
        )
        await relational.insert(TextRecord(
            node_id=summary_id, text_payload=summary_text,
            source_uri=source_uri, dataset_id=dataset_id,
        ))

    # 6. Link summary → chunks (parent-child)
    for child_id in node_ids:
        await graph.create_parent_child_edge(
            parent_id=summary_id, child_id=child_id,
        )

    # 7. Link chunks in reading order (chunk[0] → chunk[1] → chunk[2] → ...)
    for i in range(len(node_ids) - 1):
        await graph.create_sequence_edge(
            from_id=node_ids[i], to_id=node_ids[i + 1],
        )

    duration_ms = (time.time() - start) * 1000

    return IngestResult(
        chunks_created=len(new_chunks),
        nodes_created=len(new_chunks) + (0 if existing_summary else 1),
        glossary_terms_added=0,
        duration_ms=round(duration_ms, 1),
    )


async def _make_summary(chunk_texts: list[str]) -> tuple[str, str]:
    """
    Generate a summary. COST-OPTIMIZED:
    - ≤3 chunks: use the first chunk as-is (0 LLM calls)
    - >3 chunks: summarize a sample (1 LLM call)

    Returns: (summary_text, cost_note)
    """
    if len(chunk_texts) <= 3:
        # Small file — first chunk IS the summary. Zero LLM calls.
        return chunk_texts[0], "free (reused first chunk)"

    # Larger file — summarize a sample
    sample = chunk_texts[:2] + [chunk_texts[-1]]
    combined = "\n\n---\n\n".join(sample)

    # Truncate if too long — token-aware with guaranteed progress
    while count_tokens(combined) > 2000:
        new_len = len(combined) * 2000 // count_tokens(combined)
        new_len = min(new_len, len(combined) - 100)  # Ensure progress
        if new_len <= 0:
            break
        combined = combined[:new_len]

    summary = await summarize(combined)
    return summary, "1 LLM call"


def _make_node_id(content: str) -> str:
    """
    Deterministic node ID from content hash.
    Same content → same ID (deduplication).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))
