"""
Active Pathfinding — The core traversal algorithm.

COST-OPTIMIZED: Typically 1 embed + 2-4 LLM calls per query.

This is the brain of the system. Given a query, it:
1. Teleports into the graph via vector similarity (Hybrid RAG Teleporter)
2. Navigates node-by-node using LLM decision-making
3. Collects evidence, backtracks on dead ends
4. Returns found chunks or falls back to vector search (Escape Hatch)
5. Cements successful paths as Layer 2 edges (neuroplasticity)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from context_memory.agents.llm import call_llm, embed, summarize, verify, count_tokens
from context_memory.config import settings
from context_memory.pathfinding.cache import SemanticCache
from context_memory.storage.graph import GraphStore
from context_memory.storage.relational import RelationalStore
from context_memory.storage.schemas import GraphEdge
from context_memory.storage.vector import VectorStore
from context_memory.types import ContextResult

# Module-level cache singleton — persists across queries in the same process
_cache = SemanticCache(threshold=0.92, max_size=200, ttl_seconds=3600)


# ──────────────────────────────────────────────
# Traversal State
# ──────────────────────────────────────────────

@dataclass
class TraversalState:
    """All state carried through the traversal loop."""
    original_query: str
    dataset_id: str
    agent_id: str
    current_node_id: str = ""
    context_buffer: list[str] = field(default_factory=list)
    visited_nodes: set[str] = field(default_factory=set)
    target_payloads: list[str] = field(default_factory=list)
    target_sources: list[str] = field(default_factory=list)
    target_node_ids: list[str] = field(default_factory=list)
    traversal_log: list[dict] = field(default_factory=list)
    backtrack_stack: list[str] = field(default_factory=list)  # Fix #4: actual backtracking
    backtrack_count: int = 0
    hops: int = 0
    max_hops: int = 15
    llm_calls: int = 0  # Track cost


# ──────────────────────────────────────────────
# The Active Pathfinder
# ──────────────────────────────────────────────

async def find_context(
    query: str,
    dataset_id: str,
    agent_id: str,
    max_tokens: int,
    graph: GraphStore,
    vector: VectorStore,
    relational: RelationalStore,
) -> ContextResult:
    """
    Main entry. Cost budget: 1 embed + N decision calls + 1 verify.
    Typical query: 1 + 3 + 1 = 5 API calls total.
    """

    # ── Step 0: Cache check (0 API calls) ──
    # We need the vector for cache lookup, but we can check cache first
    # with a quick embed (which we'd need anyway)
    query_vector = await embed(query)

    cached = _cache.lookup(query_vector)
    if cached:
        # Clone metadata to mark as cache hit
        meta = dict(cached.traversal_metadata)
        meta["status"] = "cache_hit"
        meta["llm_calls"] = 1  # Just the embed call
        return ContextResult(
            chunks=cached.chunks,
            journey_summary="Cache hit — returning stored result.",
            confidence=cached.confidence,
            sources=cached.sources,
            traversal_metadata=meta,
        )

    # ── Step 1: Teleport — vector similarity entry ──

    teleport_results = await vector.search(
        query_vector=query_vector,
        dataset_id=dataset_id,
        top_k=3,
        layer_filter=1,  # Layer 1 summaries first
    )

    if not teleport_results:
        teleport_results = await vector.search(
            query_vector=query_vector,
            dataset_id=dataset_id,
            top_k=3,
        )

    if not teleport_results:
        return ContextResult(
            chunks=[],
            journey_summary="No data found in the dataset.",
            confidence=0.0,
            sources=[],
            traversal_metadata={"status": "empty_dataset", "llm_calls": 1},
        )

    # Broad query detection (no LLM call — just score check)
    top_score = teleport_results[0]["score"]
    if top_score < settings.broad_query_threshold:
        return await _broad_query_fallback(teleport_results, relational, max_tokens)

    entry_node_id = teleport_results[0]["id"]

    # ── Step 1.5: Shortcut-First — check learned edges BEFORE traversing ──
    # Uses Qdrant HNSW search (O(log N)) instead of Neo4j linear scan.
    # Cost: 0 LLM calls (just a DB read).
    try:
        shortcut = await vector.search_shortcuts(query_vector, dataset_id)
    except Exception:
        shortcut = None  # Graceful fallback

    if shortcut:
        # Verify the Neo4j edge exists and is not quarantined
        weight = await graph.get_edge_weight(shortcut["source_id"], shortcut["target_id"])
        if weight is not None:
            target_text = await relational.get_text(shortcut["target_id"])
            if target_text:
                # Verify it's still valid (1 LLM call)
                is_valid = await verify(query, target_text)
                if is_valid:
                    source = await relational.get_source_uri(shortcut["target_id"])
                    return ContextResult(
                        chunks=[target_text],
                        journey_summary=f"Shortcut hit (similarity={shortcut['similarity']:.2f}, weight={weight})",
                        confidence=0.9,
                        sources=[source] if source else [],
                        traversal_metadata={
                            "status": "shortcut",
                            "llm_calls": 2,  # 1 embed + 1 verify
                            "shortcut_similarity": shortcut["similarity"],
                            "shortcut_weight": weight,
                        },
                    )

    # ── Step 2: Traverse (only if no shortcut found) ──
    state = TraversalState(
        original_query=query,
        dataset_id=dataset_id,
        agent_id=agent_id,
        current_node_id=entry_node_id,
        llm_calls=1,  # Count the embed call
    )
    state.visited_nodes.add(entry_node_id)

    entry_text = await relational.get_text(entry_node_id)
    if entry_text:
        state.context_buffer.append(entry_text)

    result = await _traversal_loop(state, graph, vector, relational, query_vector)

    # ── Step 3: Cement successful paths ──
    if result.confidence > 0.0 and result.chunks:
        await _cement_path(state, query_vector, graph, vector)

    # ── Step 4: Cache the result for future similar queries ──
    _cache.store(query_vector, result)

    return result


# ──────────────────────────────────────────────
# The Traversal Loop — 1 LLM CALL PER HOP
# ──────────────────────────────────────────────

async def _traversal_loop(
    state: TraversalState,
    graph: GraphStore,
    vector: VectorStore,
    relational: RelationalStore,
    query_vector: list[float],
) -> ContextResult:
    """
    Cost-optimized loop. ONE unified LLM prompt per hop that handles:
    - "Is current node the answer?" (FOUND_TARGET)
    - "Which neighbor to visit?" (Node ID)
    - "Dead end?" (BACKTRACK)
    """

    while state.hops < state.max_hops:
        state.hops += 1

        # Fetch current node text (DB call, not LLM)
        current_text = await relational.get_text(state.current_node_id) or ""

        # Get neighbors (DB call, not LLM)
        neighbors = await graph.get_neighbors(state.current_node_id, top_k=5)
        unvisited = [n for n in neighbors if n["target_id"] not in state.visited_nodes]

        # Build neighbor info (DB calls, not LLM)
        neighbor_ids = [n["target_id"] for n in unvisited[:3]]
        neighbor_texts = await relational.get_texts_batch(neighbor_ids) if neighbor_ids else {}

        choices_block = ""
        valid_ids = []
        for nid in neighbor_ids:
            text = neighbor_texts.get(nid, "No summary available")
            choices_block += f"\n- Node {nid}: <data_begin>{text[:200]}<data_end>"
            valid_ids.append(nid)

        # ── THE ONE LLM CALL PER HOP ──
        # Unified prompt: check current + decide next in one call
        has_choices = len(valid_ids) > 0

        prompt = f"""You are navigating a knowledge graph to answer a question.

QUESTION: {state.original_query}

YOU ARE CURRENTLY AT:
<data_begin>{current_text[:400]}<data_end>

{"PATHS FORWARD:" + choices_block if has_choices else "NO PATHS FORWARD (dead end)."}

RESPOND WITH EXACTLY ONE OF:
1. FOUND_TARGET — if the text at your CURRENT location answers the question
2. {"Node <id> — the ID of the most promising path" if has_choices else "(not available — no paths)"}
3. BACKTRACK — if nothing here is relevant

ONE WORD OR NODE ID ONLY. No explanation."""

        decision = await call_llm(model=settings.summarizer_model, prompt=prompt)
        state.llm_calls += 1

        decision_clean = decision.strip().upper()

        state.traversal_log.append({
            "hop": state.hops,
            "at": state.current_node_id,
            "neighbors": neighbor_ids,
            "decision": decision.strip(),
        })

        # ── Parse decision ──

        if "FOUND_TARGET" in decision_clean:
            # Verify with one more cheap call (anti-hallucination)
            is_valid = await verify(state.original_query, current_text)
            state.llm_calls += 1

            if is_valid:
                state.target_payloads.append(current_text)
                state.target_node_ids.append(state.current_node_id)
                source = await relational.get_source_uri(state.current_node_id)
                if source:
                    state.target_sources.append(source)

                # Also grab sibling chunks for richer context (DB only, no LLM)
                siblings = await _get_sibling_chunks(
                    state.current_node_id, graph, relational, state.visited_nodes
                )
                for sib_text, sib_id, sib_source in siblings:
                    state.target_payloads.append(sib_text)
                    state.target_node_ids.append(sib_id)
                    if sib_source:
                        state.target_sources.append(sib_source)

                journey = " → ".join(log["at"][:12] for log in state.traversal_log)
                return ContextResult(
                    chunks=state.target_payloads,
                    journey_summary=f"Found in {state.hops} hops: {journey}",
                    confidence=0.85,
                    sources=state.target_sources,
                    traversal_metadata={
                        "status": "found",
                        "hops": state.hops,
                        "llm_calls": state.llm_calls,
                        "nodes_visited": len(state.visited_nodes),
                    },
                )
            else:
                # Verifier said no — current node is a false positive
                state.backtrack_count += 1

        elif "BACKTRACK" in decision_clean:
            state.backtrack_count += 1

        else:
            # Try to extract a node ID
            chosen_id = _extract_node_id(decision, valid_ids)
            if chosen_id:
                state.visited_nodes.add(chosen_id)
                state.backtrack_stack.append(state.current_node_id)  # Remember where we were
                state.current_node_id = chosen_id
                state.backtrack_count = 0

                new_text = neighbor_texts.get(chosen_id, "")
                if new_text:
                    state.context_buffer.append(new_text)

                # Fold context buffer if growing too large (1 LLM call, rarely)
                total_tokens = sum(count_tokens(c) for c in state.context_buffer)
                if total_tokens > 3000:
                    folded = await summarize("\n\n---\n\n".join(state.context_buffer))
                    state.llm_calls += 1
                    state.context_buffer = [folded]

                continue  # Skip the backtrack check below
            else:
                state.backtrack_count += 1

        # Backtrack limit check
        if state.backtrack_count > settings.max_backtrack:
            return await _escape_hatch(state, vector, relational, query_vector)

        # Fix #4: Actually backtrack to a previous node
        if state.backtrack_stack:
            state.current_node_id = state.backtrack_stack.pop()
            if state.context_buffer:
                state.context_buffer.pop()

    # Max hops exceeded
    return await _escape_hatch(state, vector, relational, query_vector)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

async def _get_sibling_chunks(
    node_id: str,
    graph: GraphStore,
    relational: RelationalStore,
    visited: set[str],
) -> list[tuple[str, str, str]]:
    """
    Get sibling chunks (same parent) for richer context.
    Follows incoming PARENT_CHILD edge up to parent, then outgoing edges down.
    All DB calls, zero LLM calls.
    """
    siblings = []

    # First try: find parent via incoming PARENT_CHILD edge
    parent_results = await graph.get_parents(node_id)
    if parent_results:
        parent_id = parent_results[0]["source_id"]
        # Get all children of the parent
        children = await graph.get_neighbors(parent_id, top_k=10)
        for c in children:
            cid = c["target_id"]
            if cid != node_id and cid not in visited:
                text = await relational.get_text(cid)
                source = await relational.get_source_uri(cid)
                if text:
                    siblings.append((text, cid, source or ""))
                if len(siblings) >= 2:
                    break
    else:
        # Fallback: grab sequential neighbors (NEXT_IN_DOC)
        neighbors = await graph.get_neighbors(node_id, top_k=5)
        for n in neighbors[:2]:
            nid = n["target_id"]
            if nid not in visited:
                text = await relational.get_text(nid)
                source = await relational.get_source_uri(nid)
                if text:
                    siblings.append((text, nid, source or ""))
    return siblings


async def _escape_hatch(
    state: TraversalState,
    vector: VectorStore,
    relational: RelationalStore,
    query_vector: list[float],
) -> ContextResult:
    """
    Fallback: vector similarity search. ZERO additional LLM calls —
    reuses the query_vector from the initial embed call.
    """
    # Reuse the query_vector — don't re-embed!
    results = await vector.search(
        query_vector=query_vector,
        dataset_id=state.dataset_id,
        top_k=5,
        layer_filter=0,
    )

    if not results:
        return ContextResult(
            chunks=["This information does not appear to exist in the dataset."],
            journey_summary="Not found.",
            confidence=0.0,
            sources=[],
            traversal_metadata={
                "status": "not_found",
                "hops": state.hops,
                "llm_calls": state.llm_calls,
            },
        )

    top_ids = [r["id"] for r in results]
    texts = await relational.get_texts_batch(top_ids)
    chunks = [texts[nid] for nid in top_ids if nid in texts]
    top_score = results[0]["score"] if results else 0.0

    return ContextResult(
        chunks=chunks[:3],
        journey_summary="Escape hatch — vector search fallback.",
        confidence=round(top_score * 0.5, 2),
        sources=[],
        traversal_metadata={
            "status": "escape_hatch",
            "hops": state.hops,
            "llm_calls": state.llm_calls,
            "vector_scores": [r["score"] for r in results[:3]],
        },
    )


async def _broad_query_fallback(
    teleport_results: list[dict],
    relational: RelationalStore,
    max_tokens: int,
) -> ContextResult:
    """
    Broad queries: return Layer 1 summaries directly.
    ZERO LLM calls — just DB reads.
    """
    top_ids = [r["id"] for r in teleport_results]
    texts = await relational.get_texts_batch(top_ids)
    chunks = [texts[nid] for nid in top_ids if nid in texts]

    return ContextResult(
        chunks=chunks,
        journey_summary="Broad query — returning summaries.",
        confidence=0.5,
        sources=[],
        traversal_metadata={"status": "broad_query", "llm_calls": 1},
    )


async def _cement_path(
    state: TraversalState,
    query_vector: list[float],
    graph: GraphStore,
    vector: VectorStore,
) -> None:
    """
    Cement a Layer 2 edge. ZERO LLM calls.
    - Neo4j: creates/reinforces SEMANTIC_LENS edge (weight, trace)
    - Qdrant: stores query vector as layer=2 shortcut (O(log N) lookup)
    """
    if not state.target_node_ids or not state.traversal_log:
        return

    entry_node_id = state.traversal_log[0]["at"]
    for target_id in state.target_node_ids:
        if entry_node_id == target_id:
            continue

        edge = GraphEdge(
            source_id=entry_node_id,
            target_id=target_id,
            edge_type="semantic_lens",
            procedural_trace=[log["at"] for log in state.traversal_log],
            created_by_agent=state.agent_id,
        )
        await graph.cement_edge(edge)

        # Store shortcut vector in Qdrant (layer=2) for O(log N) search
        shortcut_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"shortcut:{entry_node_id}->{target_id}"
        ))
        await vector.upsert_shortcut(
            shortcut_id=shortcut_id,
            vector=query_vector,
            dataset_id=state.dataset_id,
            source_id=entry_node_id,
            target_id=target_id,
        )


def _extract_node_id(decision: str, valid_ids: list[str]) -> str | None:
    """
    Extract and validate a node ID from the LLM's response.
    Only returns IDs that exist in the current valid neighborhood.
    """
    for valid_id in valid_ids:
        if valid_id in decision:
            return valid_id
    return None
