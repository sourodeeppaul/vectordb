"""
Example usage of the HNSWIndex.
"""

import numpy as np
import time
from vectordb.index import HNSWIndex, FlatIndex


def main():
    print("=" * 60)
    print("HNSW Index Usage Example")
    print("=" * 60)
    
    # Configuration
    dimension = 128
    n_vectors = 50000
    n_queries = 100
    k = 10
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {dimension}")
    print(f"  Vectors: {n_vectors:,}")
    print(f"  Queries: {n_queries}")
    print(f"  K: {k}")
    
    # Generate data
    print("\n1. Generating random vectors...")
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    
    # Create HNSW index
    print("\n2. Building HNSW index...")
    hnsw = HNSWIndex(
        dimension=dimension,
        metric="euclidean",
        M=16,
        ef_construction=200,
        ef_search=50,
        seed=42,
    )
    
    start = time.time()
    for i in range(n_vectors):
        hnsw.add(f"vec{i}", vectors[i], {"index": i})
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start
            print(f"   Added {i+1:,} vectors ({(i+1)/elapsed:.0f} vec/s)")
    
    build_time = time.time() - start
    print(f"   Total build time: {build_time:.2f}s")
    
    # Get stats
    print("\n3. Index statistics:")
    stats = hnsw.stats()
    print(f"   Vectors: {stats.vector_count:,}")
    print(f"   Memory: {stats.memory_bytes / 1024 / 1024:.2f} MB")
    print(f"   Max level: {stats.extra['max_level']}")
    print(f"   Avg connections: {stats.extra['avg_connections']:.1f}")
    
    # Search benchmark
    print("\n4. Search benchmark...")
    
    ef_values = [10, 25, 50, 100, 200]
    
    print(f"\n   {'ef':>6} | {'QPS':>8} | {'Latency':>10}")
    print("   " + "-" * 32)
    
    for ef in ef_values:
        hnsw.set_ef_search(ef)
        
        start = time.time()
        for q in queries:
            _ = hnsw.search(q, k=k)
        elapsed = time.time() - start
        
        qps = n_queries / elapsed
        latency_ms = elapsed / n_queries * 1000
        
        print(f"   {ef:>6} | {qps:>8.0f} | {latency_ms:>8.2f}ms")
    
    # Compare with flat index (for recall measurement)
    print("\n5. Recall comparison with brute-force...")
    
    # Build flat index with subset for speed
    subset_size = min(10000, n_vectors)
    flat = FlatIndex(dimension=dimension, metric="euclidean")
    for i in range(subset_size):
        flat.add(f"vec{i}", vectors[i])
    
    # Rebuild HNSW with subset
    hnsw_small = HNSWIndex(dimension=dimension, M=16, ef_construction=200, seed=42)
    for i in range(subset_size):
        hnsw_small.add(f"vec{i}", vectors[i])
    
    # Measure recall
    test_queries = queries[:20]
    
    print(f"\n   {'ef':>6} | {'Recall@10':>10}")
    print("   " + "-" * 22)
    
    for ef in [10, 25, 50, 100, 200]:
        recalls = []
        
        for q in test_queries:
            hnsw_results = hnsw_small.search(q, k=k, ef=ef)
            flat_results = flat.search(q, k=k)
            
            hnsw_ids = {r.id for r in hnsw_results}
            flat_ids = {r.id for r in flat_results}
            
            recall = len(hnsw_ids & flat_ids) / k
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        print(f"   {ef:>6} | {avg_recall:>10.1%}")
    
    # Filtered search
    print("\n6. Filtered search...")
    
    # Add metadata
    hnsw_filtered = HNSWIndex(dimension=32, M=8, seed=42)
    categories = ["electronics", "clothing", "books", "home", "sports"]
    
    for i in range(1000):
        hnsw_filtered.add(
            f"item{i}",
            np.random.randn(32).astype(np.float32),
            {"category": categories[i % 5], "price": float(10 + i % 100)}
        )
    
    query = np.random.randn(32).astype(np.float32)
    
    # Unfiltered
    results = hnsw_filtered.search(query, k=5)
    print("\n   Unfiltered results:")
    for r in results:
        print(f"   - {r.id}: {r.metadata['category']}, ${r.metadata['price']:.0f}")
    
    # Filtered by category
    results = hnsw_filtered.search(
        query, k=5,
        filter_fn=lambda id, meta: meta.get("category") == "electronics"
    )
    print("\n   Filtered (electronics only):")
    for r in results:
        print(f"   - {r.id}: {r.metadata['category']}, ${r.metadata['price']:.0f}")
    
    # Serialization
    print("\n7. Serialization...")
    
    data = hnsw_filtered.to_dict()
    restored = HNSWIndex.from_dict(data)
    
    print(f"   Original: {hnsw_filtered.size} vectors")
    print(f"   Restored: {restored.size} vectors")
    
    # Verify search works
    orig_results = hnsw_filtered.search(query, k=3)
    rest_results = restored.search(query, k=3)
    
    orig_ids = [r.id for r in orig_results]
    rest_ids = [r.id for r in rest_results]
    
    print(f"   Original top-3: {orig_ids}")
    print(f"   Restored top-3: {rest_ids}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()