"""
Minimal smoke test — ingest a single small file to verify the full pipeline.
Run: python -m tests.smoke_test
"""

import asyncio
from pathlib import Path
from context_memory.api import ingest, health, close


# Create a small test document inline
TEST_DOC = """\
## What Is Context Memory?

The Context Memory Framework is a neuroplastic AI memory module.
It organizes information into a 3-layer graph:
- Layer 0: Raw data chunks
- Layer 1: Summaries and indexes
- Layer 2: Learned shortcuts (Ephemeral Lenses)

## How It Works

When an agent asks a question, the Pathfinder traverses the graph
using Active Pathfinding — like navigating a city using known shortcuts
instead of reading every street sign.

## Why It Matters

Traditional RAG dumps every chunk into a flat vector space.
Context Memory builds structure, learns from queries, and gets faster over time.
"""


async def main():
    print("=" * 50)
    print("Context Memory Framework — Smoke Test")
    print("=" * 50)

    # Write test doc to a temp file
    test_dir = Path("./test_data")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_doc.md"
    test_file.write_text(TEST_DOC, encoding="utf-8")

    print(f"\nIngesting: {test_file}")
    print("(using 3s delay between embeddings for free tier)\n")

    try:
        result = await ingest(
            source_uri=str(test_file),
            dataset_id="test",
            embed_delay=3.0,
        )

        print(f"--- Ingest Results ---")
        print(f"Chunks created: {result.chunks_created}")
        print(f"Nodes created:  {result.nodes_created}")
        print(f"Duration:       {result.duration_ms:.0f}ms")

        # Check health
        report = await health()
        print(f"\n--- Health ---")
        print(f"Total nodes:    {report.total_nodes}")
        print(f"Total edges:    {report.total_edges}")
        print(f"Total datasets: {report.total_datasets}")

        print("\n✓ Pipeline works end-to-end!")

    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        raise
    finally:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
