"""
Neo4j graph database wrapper.
Handles all Cypher queries for nodes and edges.
Uses AsyncGraphDatabase for non-blocking operations.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from neo4j import AsyncGraphDatabase

from context_memory.config import settings
from context_memory.storage.schemas import GraphNode, GraphEdge


class GraphStore:
    """Async Neo4j wrapper. All graph reads and writes go through here."""

    _indexes_checked: bool = False  # Class-level: only check once per process

    def __init__(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def ensure_indexes(self) -> None:
        """Create indexes for fast lookups. Skips if already checked this process."""
        if GraphStore._indexes_checked:
            return
        async with self._driver.session() as session:
            # Check which indexes already exist
            result = await session.run("SHOW INDEXES YIELD name RETURN name")
            existing = {record["name"] async for record in result}

            if "node_id_index" not in existing:
                await session.run(
                    "CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.node_id)"
                )
            if "dataset_id_index" not in existing:
                await session.run(
                    "CREATE INDEX dataset_id_index IF NOT EXISTS FOR (n:Node) ON (n.dataset_id)"
                )
        GraphStore._indexes_checked = True

    # ── Node Operations ──────────────────────────

    async def create_node(self, node: GraphNode) -> None:
        """Create or update a graph node. Uses MERGE to prevent duplicates."""
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (n:Node {node_id: $node_id})
                ON CREATE SET
                    n.layer = $layer,
                    n.node_type = $node_type,
                    n.dataset_id = $dataset_id,
                    n.last_accessed = datetime($last_accessed)
                ON MATCH SET
                    n.last_accessed = datetime($last_accessed)
                """,
                node_id=node.node_id,
                layer=node.layer,
                node_type=node.node_type,
                dataset_id=node.dataset_id,
                last_accessed=node.last_accessed.isoformat(),
            )

    async def bulk_create_nodes(self, nodes: list[dict], batch_size: int = 5000) -> None:
        """
        Fast bulk ingestion of nodes via UNWIND.
        nodes shape: [{"node_id": x, "layer": y, "node_type": z, "dataset_id": w}, ...]
        """
        async with self._driver.session() as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                await session.run(
                    """
                    UNWIND $batch AS item
                    MERGE (n:Node {node_id: item.node_id})
                    ON CREATE SET
                        n.layer = item.layer,
                        n.node_type = item.node_type,
                        n.dataset_id = CASE WHEN item.dataset_id IS NOT NULL THEN item.dataset_id ELSE 'default' END,
                        n.last_accessed = datetime()
                    ON MATCH SET
                        n.last_accessed = datetime()
                    """,
                    batch=batch
                )

    async def get_neighbors(self, node_id: str, top_k: int = 3) -> list[dict]:
        """
        Find the top-K connected nodes sorted by effective weight.
        Returns edge + target node info for the Pathfinder's decision prompt.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (n:Node {node_id: $node_id})-[r]->(target:Node)
                RETURN target.node_id AS target_id,
                       target.layer AS layer,
                       r.base_weight AS base_weight,
                       r.edge_type AS edge_type,
                       r.last_traversed AS last_traversed,
                       r.created_by_agent AS created_by_agent,
                       r.quarantine AS quarantine,
                       r.context_vector AS context_vector
                ORDER BY r.base_weight DESC
                LIMIT $top_k
                """,
                node_id=node_id,
                top_k=top_k,
            )
            return [dict(record) async for record in result]

    async def get_parents(self, node_id: str) -> list[dict]:
        """Find parent nodes via incoming PARENT_CHILD edges."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (parent:Node)-[r:PARENT_CHILD]->(n:Node {node_id: $node_id})
                RETURN parent.node_id AS source_id, parent.layer AS layer
                LIMIT 3
                """,
                node_id=node_id,
            )
            return [dict(record) async for record in result]

    # ── Edge Operations ──────────────────────────

    async def cement_edge(self, edge: GraphEdge) -> None:
        """
        Create or reinforce a Layer 2 Semantic Lens edge.
        - ON CREATE: weight=1.0, quarantine=true, stores agent_id
        - ON MATCH:  pushes version snapshot as string, increments weight
        NOTE: context_vector is stored in Qdrant (layer=2) for O(log N) lookup.
        """
        ceiling = settings.edge_weight_ceiling
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (start:Node {node_id: $source_id}), (end:Node {node_id: $target_id})
                MERGE (start)-[r:SEMANTIC_LENS]->(end)
                ON CREATE SET
                    r.base_weight = 1.0,
                    r.procedural_trace = $procedural_trace,
                    r.last_traversed = datetime(),
                    r.version_history = [],
                    r.created_by_agent = $agent_id,
                    r.quarantine = true
                ON MATCH SET
                    r.version_history = r.version_history[-2..] + [
                        toString(r.base_weight) + '|' + toString(r.last_traversed)
                    ],
                    r.base_weight = CASE
                        WHEN r.base_weight + 0.2 > $ceiling THEN $ceiling
                        ELSE r.base_weight + 0.2
                    END,
                    r.last_traversed = datetime(),
                    r.quarantine = CASE
                        WHEN r.base_weight >= 2.0 THEN false
                        ELSE r.quarantine
                    END
                """,
                source_id=edge.source_id,
                target_id=edge.target_id,
                procedural_trace=edge.procedural_trace,
                agent_id=edge.created_by_agent,
                ceiling=ceiling,
            )

    async def create_parent_child_edge(
        self, parent_id: str, child_id: str
    ) -> None:
        """Create a structural parent→child edge (Layer 1 → Layer 0)."""
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (p:Node {node_id: $parent_id}), (c:Node {node_id: $child_id})
                MERGE (p)-[r:PARENT_CHILD]->(c)
                ON CREATE SET r.base_weight = 1.0, r.edge_type = 'parent_child'
                """,
                parent_id=parent_id,
                child_id=child_id,
            )

    async def create_sequence_edge(
        self, from_id: str, to_id: str
    ) -> None:
        """Create a NEXT_IN_DOC edge preserving reading order between chunks."""
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (a:Node {node_id: $from_id}), (b:Node {node_id: $to_id})
                MERGE (a)-[r:NEXT_IN_DOC]->(b)
                ON CREATE SET r.base_weight = 0.5, r.edge_type = 'sequence'
                """,
                from_id=from_id,
                to_id=to_id,
            )

    async def bulk_create_topology(
        self, edges: list[dict], rel_type: str = "IMPLIES", batch_size: int = 5000
    ) -> None:
        """
        Fast bulk ingestion of raw topological edges via UNWIND.
        edges shape: [{"source_id": x, "target_id": y}, ...]
        """
        # Basic sanitization for cypher string formatting
        import re
        if not re.match(r"^[A-Za-z0-9_]+$", rel_type):
            raise ValueError(f"Invalid relationship type: {rel_type}")
            
        async with self._driver.session() as session:
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                await session.run(
                    f"""
                    UNWIND $batch AS item
                    MATCH (a:Node {{node_id: item.source_id}}), (b:Node {{node_id: item.target_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    ON CREATE SET r.base_weight = 1.0, r.edge_type = 'topology'
                    """,
                    batch=batch
                )

    async def get_edge_weight(
        self, source_id: str, target_id: str
    ) -> float | None:
        """Get the weight of SEMANTIC_LENS edge (for shortcut verification)."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Node {node_id: $source_id})-[r:SEMANTIC_LENS]->(t:Node {node_id: $target_id})
                WHERE r.quarantine = false
                RETURN r.base_weight AS weight
                """,
                source_id=source_id,
                target_id=target_id,
            )
            record = await result.single()
            return record["weight"] if record else None

    async def apply_confidence_decay(
        self, node_id: str, multiplier: float = 0.5
    ) -> None:
        """Multiply all incoming Layer 2 edge weights by a decay factor."""
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH ()-[r:SEMANTIC_LENS]->(n:Node {node_id: $node_id})
                SET r.base_weight = r.base_weight * $multiplier
                """,
                node_id=node_id,
                multiplier=multiplier,
            )

    # ── Stats ────────────────────────────────────

    async def count_nodes(self) -> int:
        async with self._driver.session() as session:
            result = await session.run("MATCH (n:Node) RETURN count(n) AS c")
            record = await result.single()
            return record["c"] if record else 0

    async def count_edges(self) -> int:
        async with self._driver.session() as session:
            result = await session.run("MATCH ()-[r]->() RETURN count(r) AS c")
            record = await result.single()
            return record["c"] if record else 0

    async def count_datasets(self) -> int:
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (n:Node) RETURN count(DISTINCT n.dataset_id) AS c"
            )
            record = await result.single()
            return record["c"] if record else 0
