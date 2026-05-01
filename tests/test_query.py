"""
End-to-end query test — ingest a doc, then query against it.
Run: python -m tests.test_query
"""

import asyncio
from context_memory.api import ingest, query, close


async def main():
    print("=" * 60)
    print("Context Memory Framework — Query Test")
    print("=" * 60)

    # Step 1: Ingest a test document (skip if already ingested)
    print("\n[1/3] Ingesting test data...")
    result = await ingest(
        source_uri="./test_data/test_doc.md",
        dataset_id="test",
        embed_delay=3.0,
    )
    print(f"      Chunks: {result.chunks_created}, Nodes: {result.nodes_created}")

    # Step 2: Query
    print("\n[2/3] Querying: 'How does the pathfinding work?'")
    ctx = await query(
        query_text="How does the pathfinding work?",
        dataset_id="test",
    )

    print(f"\n--- Query Result ---")
    print(f"Status:     {ctx.traversal_metadata.get('status', 'unknown')}")
    print(f"Confidence: {ctx.confidence}")
    print(f"Hops:       {ctx.traversal_metadata.get('hops', 'N/A')}")
    print(f"Chunks:     {len(ctx.chunks)}")
    if ctx.chunks:
        print(f"\nFirst chunk preview:")
        print(f"  {ctx.chunks[0][:200]}...")
    print(f"\nJourney: {ctx.journey_summary[:200]}")

    # Step 3: Another query
    print("\n[3/3] Querying: 'What is context memory?'")
    ctx2 = await query(
        query_text="What is context memory?",
        dataset_id="test",
    )

    print(f"\n--- Query Result ---")
    print(f"Status:     {ctx2.traversal_metadata.get('status', 'unknown')}")
    print(f"Confidence: {ctx2.confidence}")
    print(f"Chunks:     {len(ctx2.chunks)}")
    if ctx2.chunks:
        print(f"\nFirst chunk preview:")
        print(f"  {ctx2.chunks[0][:200]}...")

    await close()
    print("\n✓ Done.")


if __name__ == "__main__":
    asyncio.run(main())
