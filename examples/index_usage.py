"""
Example usage of the FlatIndex.
"""

import numpy as np
import time
from vectordb.index import FlatIndex, create_index


def main():
    print("=" * 60)
    print("FlatIndex Usage Example")
    print("=" * 60)
    
    # 1. Create index
    print("\n1. Creating index...")
    index = FlatIndex(
        dimension=128,
        metric="cosine",
        normalize=True,
    )
    print(f"   Created: {index}")
    
    # 2. Add vectors
    print("\n2. Adding vectors...")
    
    # Single add
    index.add(
        "doc_001",
        np.random.randn(128),
        {"title": "Introduction", "category": "tutorial"}
    )
    
    # Batch add
    n_vectors = 10000
    ids = [f"doc_{i:05d}" for i in range(2, n_vectors + 2)]
    vectors = np.random.randn(n_vectors, 128).astype(np.float32)
    metadata = [
        {"title": f"Document {i}", "category": ["tutorial", "guide", "reference"][i % 3]}
        for i in range(n_vectors)
    ]
    
    start = time.time()
    count = index.add_batch(ids, vectors, metadata)
    add_time = time.time() - start
    
    print(f"   Added {count} vectors in {add_time:.3f}s")
    print(f"   Total: {index.size} vectors")
    
    # 3. Search
    print("\n3. Basic search...")
    query = np.random.randn(128).astype(np.float32)
    
    start = time.time()
    results = index.search(query, k=5)
    search_time = time.time() - start
    
    print(f"   Found {len(results)} results in {search_time*1000:.2f}ms")
    for r in results:
        print(f"   - {r.id}: distance={r.distance:.4f}, score={r.score:.4f}")
    
    # 4. Filtered search
    print("\n4. Filtered search (category='tutorial')...")
    
    def filter_fn(id, metadata):
        return metadata.get("category") == "tutorial"
    
    start = time.time()
    results = index.search(query, k=5, filter_fn=filter_fn)
    search_time = time.time() - start
    
    print(f"   Found {len(results)} results in {search_time*1000:.2f}ms")
    for r in results:
        print(f"   - {r.id}: category={r.metadata.get('category')}")
    
    # 5. Range search
    print("\n5. Range search (radius=0.5)...")
    
    results = index.range_search(query, radius=0.5, max_results=10)
    print(f"   Found {len(results)} vectors within radius")
    
    # 6. Search by ID
    print("\n6. Search by ID (find similar to doc_001)...")
    
    results = index.search_by_id("doc_001", k=5)
    print(f"   Vectors similar to doc_001:")
    for r in results:
        print(f"   - {r.id}: distance={r.distance:.4f}")
    
    # 7. Batch search
    print("\n7. Batch search (10 queries)...")
    
    queries = np.random.randn(10, 128).astype(np.float32)
    
    start = time.time()
    batch_results = index.search_batch(queries, k=5)
    batch_time = time.time() - start
    
    print(f"   Processed 10 queries in {batch_time*1000:.2f}ms")
    print(f"   Average: {batch_time*100:.2f}ms per query")
    
    # 8. Update
    print("\n8. Update operations...")
    
    new_vector = np.random.randn(128).astype(np.float32)
    index.update("doc_001", vector=new_vector, metadata={"updated": True})
    
    _, meta = index.get("doc_001")
    print(f"   doc_001 updated: {meta}")
    
    # 9. Remove
    print("\n9. Remove operations...")
    
    removed = index.remove("doc_001")
    print(f"   Removed doc_001: {removed}")
    
    removed_count = index.remove_batch(["doc_00002", "doc_00003", "nonexistent"])
    print(f"   Batch removed: {removed_count}")
    
    # 10. Statistics
    print("\n10. Index statistics...")
    
    stats = index.stats()
    print(f"   Type: {stats.index_type}")
    print(f"   Vectors: {stats.vector_count}")
    print(f"   Dimension: {stats.dimension}")
    print(f"   Metric: {stats.metric}")
    print(f"   Memory: {stats.memory_bytes / 1024 / 1024:.2f} MB")
    
    # 11. Benchmark
    print("\n11. Search benchmark...")
    
    n_queries = 100
    queries = np.random.randn(n_queries, 128).astype(np.float32)
    
    start = time.time()
    for q in queries:
        _ = index.search(q, k=10)
    total_time = time.time() - start
    
    qps = n_queries / total_time
    print(f"   {n_queries} queries in {total_time:.3f}s")
    print(f"   QPS: {qps:.1f}")
    print(f"   Latency: {total_time/n_queries*1000:.2f}ms per query")
    
    # 12. Serialization
    print("\n12. Serialization...")
    
    data = index.to_dict()
    restored = FlatIndex.from_dict(data)
    
    print(f"   Original size: {index.size}")
    print(f"   Restored size: {restored.size}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()