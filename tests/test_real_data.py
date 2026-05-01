"""
Real-data test — ingest the framework's design docs, then query against them.
Tests: multi-file ingestion, cross-document traversal, edge cementing, cost tracking.

Run: python -m tests.test_real_data
"""

import asyncio
import time
from context_memory.api import ingest, query, health, close


ARTIFACTS_DIR = r"C:\Users\SANYAM\.gemini\antigravity\brain\f28b1998-64a2-4738-95b4-61d17061bd7e"

# Diverse queries to test different traversal paths
QUERIES = [
    "What is the drift-aware decay model?",
    "How does the security audit address prompt injection?",
    "What are the estimated costs and build timeline?",
    "How are edges cemented after a successful traversal?",
    "What databases does the framework use?",
]


async def main():
    print("=" * 60)
    print("Context Memory — Real Data Test")
    print("=" * 60)

    # Step 1: Ingest all design docs
    print(f"\n[1] Ingesting design docs from artifacts...")
    print("    (3s delay between embeddings for free tier)\n")

    t0 = time.time()
    result = await ingest(
        source_uri=ARTIFACTS_DIR,
        dataset_id="framework-docs",
        embed_delay=3.0,
    )
    ingest_time = time.time() - t0

    print(f"\n    Ingest complete:")
    print(f"    Chunks:    {result.chunks_created}")
    print(f"    Nodes:     {result.nodes_created}")
    print(f"    Duration:  {ingest_time:.0f}s")

    # Step 2: Health check
    report = await health()
    print(f"\n[2] Health:")
    print(f"    Nodes:    {report.total_nodes}")
    print(f"    Edges:    {report.total_edges}")
    print(f"    Datasets: {report.total_datasets}")

    # Step 3: Run queries
    print(f"\n[3] Running {len(QUERIES)} queries...\n")

    total_calls = 0
    for i, q in enumerate(QUERIES, 1):
        print(f"  Q{i}: {q}")
        t0 = time.time()
        ctx = await query(q, dataset_id="framework-docs")
        dur = time.time() - t0
        calls = ctx.traversal_metadata.get("llm_calls", "?")
        total_calls += calls if isinstance(calls, int) else 0

        print(f"      Status:     {ctx.traversal_metadata.get('status', '?')}")
        print(f"      Confidence: {ctx.confidence}")
        print(f"      Hops:       {ctx.traversal_metadata.get('hops', '?')}")
        print(f"      LLM calls:  {calls}")
        print(f"      Duration:   {dur:.1f}s")
        print(f"      Chunks:     {len(ctx.chunks)}")
        if ctx.chunks:
            preview = ctx.chunks[0][:120].replace('\n', ' ')
            print(f"      Preview:    {preview}...")
        print()

    # Step 4: Check for cemented edges
    report2 = await health()
    new_edges = report2.total_edges - report.total_edges
    print(f"[4] Summary:")
    print(f"    Queries run:     {len(QUERIES)}")
    print(f"    Total LLM calls: {total_calls}")
    print(f"    Avg calls/query: {total_calls / len(QUERIES):.1f}")
    print(f"    New edges:       {new_edges} (cemented from successful queries)")
    print(f"    Total edges now: {report2.total_edges}")

    await close()
    print("\n✓ Done.")


if __name__ == "__main__":
    asyncio.run(main())
