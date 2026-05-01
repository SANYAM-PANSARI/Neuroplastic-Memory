import asyncio
import time
from context_memory.storage.graph import GraphStore

async def test_bulk_ingestion():
    graph = GraphStore()
    await graph.ensure_indexes()
    
    # Generate 1000 nodes and 1000 edges for a quick benchmark
    test_dataset = "bulk_test_dataset"
    nodes = [
        {
            "node_id": f"test_node_{i}",
            "layer": 0,
            "node_type": "Law",
            "dataset_id": test_dataset
        }
        for i in range(1000)
    ]
    
    edges = [
        {
            "source_id": f"test_node_{i}",
            "target_id": f"test_node_{min(i+1, 999)}"
        }
        for i in range(1000)
    ]
    
    print("Testing bulk_create_nodes...")
    start = time.time()
    await graph.bulk_create_nodes(nodes, batch_size=5000)
    node_time = time.time() - start
    print(f"Nodes created in {node_time:.4f}s")
    
    print("Testing bulk_create_topology...")
    start = time.time()
    await graph.bulk_create_topology(edges, rel_type="IMPLIES", batch_size=5000)
    edge_time = time.time() - start
    print(f"Edges created in {edge_time:.4f}s")
    
    # Verify count
    async with graph._driver.session() as session:
        result = await session.run(f"MATCH (n:Node {{dataset_id: '{test_dataset}'}}) RETURN count(n) AS c")
        c = (await result.single())["c"]
        print(f"Verified Node Count: {c}")
        
        result2 = await session.run(f"MATCH (n:Node {{dataset_id: '{test_dataset}'}})-[r:IMPLIES]->() RETURN count(r) AS c")
        c2 = (await result2.single())["c"]
        print(f"Verified Edge Count: {c2}")
        
        # Cleanup
        print("Cleaning up test data...")
        await session.run(f"MATCH (n:Node {{dataset_id: '{test_dataset}'}}) DETACH DELETE n")

    await graph.close()

if __name__ == "__main__":
    asyncio.run(test_bulk_ingestion())
