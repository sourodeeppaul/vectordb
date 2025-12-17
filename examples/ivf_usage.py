"""
Example usage of the IVFIndex.
"""

import numpy as np
import time
from vectordb.index import IVFIndex, FlatIndex


def main():
    print("=" * 60)
    print("IVF Index Usage Example")
    print("=" * 60)
    
    # Configuration
    dimension = 128
    n_training = 10000
    n_vectors = 100000
    n_queries = 100
    k = 10
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {dimension}")
    print(f"  Training vectors: {n_training:,}")
    print(f"  Total vectors: {n_vectors:,}")
    print(f"  Queries: {n_queries}")
    print(f"  K: {k}")
    
    # Generate data
    print("\n1. Generating random vectors...")
    np.random.seed(42)
    training_vectors = np.random.randn(n_training, dimension).astype(np.float32)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    
    # Determine number of clusters (rule of thumb: 4*sqrt(n))
    n_clusters = int(4 * np.sqrt(n_vectors))
    print(f"\n2. Creating IVF index with {n_clusters} clusters...")
    
    ivf = IVFIndex(
        dimension=dimension,
        metric="euclidean",
        n_clusters=n_clusters,
        n_probe=10,
        seed=42,
        verbose=True,
    )
    
    # Train
    print("\n3. Training index...")
    start = time.time()
    ivf.train(training_vectors)
    train_time = time.time() - start
    print(f"   Training time: {train_time:.2f}s")
    
    # Add vectors
    print("\n4. Adding vectors...")
    start = time.time()
    
    batch_size = 10000
    for i in range(0, n_vectors, batch_size):
        end = min(i + batch_size, n_vectors)
        ids = [f"vec{j}" for j in range(i, end)]
        ivf.add_batch(ids, vectors[i:end])
        
        if (i + batch_size) % 50000 == 0 or end == n_vectors:
            elapsed = time.time() - start
            print(f"   Added {end:,} vectors ({end/elapsed:.0f} vec/s)")
    
    add_time = time.time() - start
    print(f"   Total add time: {add_time:.2f}s")
    
    # Get cluster info
    print("\n5. Cluster information:")
    info = ivf.get_cluster_info()
    print(f"   Total clusters: {info['n_clusters']}")
    print(f"   Total vectors: {info['total_vectors']:,}")
    print(f"   Cluster size - min: {info['size_distribution']['min']}, "
          f"max: {info['size_distribution']['max']}, "
          f"mean: {info['size_distribution']['mean']:.1f}")
    print(f"   Empty clusters: {info['empty_clusters']}")
    print(f"   Imbalance ratio: {info['imbalance_ratio']:.2f}")
    
    # Search benchmark with different n_probe
    print("\n6. Search benchmark (varying n_probe):")
    print(f"\n   {'n_probe':>8} | {'QPS':>8} | {'Latency':>10}")
    print("   " + "-" * 35)
    
    n_probe_values = [1, 5, 10, 20, 50, 100]
    
    for n_probe in n_probe_values:
        ivf.set_n_probe(n_probe)
        
        start = time.time()
        for q in queries:
            _ = ivf.search(q, k=k)
        elapsed = time.time() - start
        
        qps = n_queries / elapsed
        latency_ms = elapsed / n_queries * 1000
        
        print(f"   {n_probe:>8} | {qps:>8.0f} | {latency_ms:>8.2f}ms")
    
    # Recall comparison (using subset for flat index)
    print("\n7. Recall comparison (using 10K subset for ground truth):")
    
    subset_size = min(10000, n_vectors)
    flat = FlatIndex(dimension=dimension, metric="euclidean")
    for i in range(subset_size):
        flat.add(f"vec{i}", vectors[i])
    
    # Rebuild IVF with subset
    ivf_small = IVFIndex(dimension=dimension, n_clusters=100, seed=42)
    ivf_small.train(vectors[:1000])
    for i in range(subset_size):
        ivf_small.add(f"vec{i}", vectors[i])
    
    test_queries = queries[:20]
    
    print(f"\n   {'n_probe':>8} | {'Recall@10':>10}")
    print("   " + "-" * 25)
    
    for n_probe in [1, 5, 10, 20, 50]:
        recalls = []
        
        for q in test_queries:
            ivf_results = ivf_small.search(q, k=k, n_probe=n_probe)
            flat_results = flat.search(q, k=k)
            
            ivf_ids = {r.id for r in ivf_results}
            flat_ids = {r.id for r in flat_results}
            
            recall = len(ivf_ids & flat_ids) / k
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        print(f"   {n_probe:>8} | {avg_recall:>10.1%}")
    
    # Filtered search
    print("\n8. Filtered search example...")
    
    ivf_filtered = IVFIndex(dimension=32, n_clusters=10, seed=42)
    ivf_filtered.train(np.random.randn(200, 32).astype(np.float32))
    
    categories = ["electronics", "clothing", "books", "home", "sports"]
    for i in range(1000):
        ivf_filtered.add(
            f"item{i}",
            np.random.randn(32).astype(np.float32),
            {"category": categories[i % 5], "price": float(10 + i % 100)}
        )
    
    query = np.random.randn(32).astype(np.float32)
    
    # Unfiltered
    results = ivf_filtered.search(query, k=5)
    print("\n   Unfiltered results:")
    for r in results:
        print(f"   - {r.id}: {r.metadata['category']}, ${r.metadata['price']:.0f}")
    
    # Filtered
    results = ivf_filtered.search(
        query, k=5,
        filter_fn=lambda id, meta: meta.get("category") == "electronics"
    )
    print("\n   Filtered (electronics only):")
    for r in results:
        print(f"   - {r.id}: {r.metadata['category']}, ${r.metadata['price']:.0f}")
    
    # Stats
    print("\n9. Index statistics:")
    stats = ivf.stats()
    print(f"   Vectors: {stats.vector_count:,}")
    print(f"   Memory: {stats.memory_bytes / 1024 / 1024:.1f} MB")
    print(f"   Clusters: {stats.extra['n_clusters']}")
    print(f"   n_probe: {stats.extra['n_probe']}")
    
    # Serialization
    print("\n10. Serialization test...")
    
    small_ivf = IVFIndex(dimension=16, n_clusters=5, seed=42)
    small_ivf.train(np.random.randn(100, 16).astype(np.float32))
    for i in range(50):
        small_ivf.add(f"vec{i}", np.random.randn(16).astype(np.float32))
    
    data = small_ivf.to_dict()
    restored = IVFIndex.from_dict(data)
    
    print(f"   Original: {small_ivf.size} vectors, trained={small_ivf.is_trained()}")
    print(f"   Restored: {restored.size} vectors, trained={restored.is_trained()}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()