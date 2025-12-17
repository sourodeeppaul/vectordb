"""
Compare Flat vs HNSW vs IVF index performance.
"""

import numpy as np
import time
from vectordb.index import FlatIndex, HNSWIndex, IVFIndex


def run_benchmark():
    print("=" * 70)
    print("Index Comparison: Flat vs HNSW vs IVF")
    print("=" * 70)
    
    # Test configurations
    configs = [
        {"n_vectors": 5000, "dimension": 128},
        {"n_vectors": 20000, "dimension": 128},
        {"n_vectors": 50000, "dimension": 128},
    ]
    
    n_queries = 100
    k = 10
    
    for config in configs:
        n_vectors = config["n_vectors"]
        dimension = config["dimension"]
        
        print(f"\n{'='*70}")
        print(f"Dataset: {n_vectors:,} vectors, {dimension} dimensions")
        print("=" * 70)
        
        # Generate data
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        queries = np.random.randn(n_queries, dimension).astype(np.float32)
        
        results = {}
        build_times = {}
        search_times = {}
        
        # =====================================================
        # Flat index
        # =====================================================
        print("\n[Flat Index]")
        flat = FlatIndex(dimension=dimension, metric="euclidean")
        
        start = time.time()
        for i in range(n_vectors):
            flat.add(f"vec{i}", vectors[i])
        build_times["flat"] = time.time() - start
        print(f"  Build: {build_times['flat']:.3f}s")
        
        start = time.time()
        flat_results = [flat.search(q, k=k) for q in queries]
        search_times["flat"] = time.time() - start
        flat_qps = n_queries / search_times["flat"]
        print(f"  Search: {flat_qps:.0f} QPS ({search_times['flat']/n_queries*1000:.2f}ms/query)")
        
        results["flat"] = flat_results
        
        # =====================================================
        # HNSW index
        # =====================================================
        print("\n[HNSW Index (M=16, ef=50)]")
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
            hnsw.add(f"vec{i}", vectors[i])
        build_times["hnsw"] = time.time() - start
        print(f"  Build: {build_times['hnsw']:.3f}s")
        
        start = time.time()
        hnsw_results = [hnsw.search(q, k=k) for q in queries]
        search_times["hnsw"] = time.time() - start
        hnsw_qps = n_queries / search_times["hnsw"]
        print(f"  Search: {hnsw_qps:.0f} QPS ({search_times['hnsw']/n_queries*1000:.2f}ms/query)")
        
        results["hnsw"] = hnsw_results
        
        # =====================================================
        # IVF index
        # =====================================================
        n_clusters = int(4 * np.sqrt(n_vectors))
        n_probe = max(1, n_clusters // 10)
        
        print(f"\n[IVF Index (clusters={n_clusters}, n_probe={n_probe})]")
        ivf = IVFIndex(
            dimension=dimension,
            metric="euclidean",
            n_clusters=n_clusters,
            n_probe=n_probe,
            seed=42,
        )
        
        start = time.time()
        ivf.train(vectors[:min(10000, n_vectors)])
        train_time = time.time() - start
        
        start = time.time()
        for i in range(n_vectors):
            ivf.add(f"vec{i}", vectors[i])
        add_time = time.time() - start
        build_times["ivf"] = train_time + add_time
        print(f"  Train: {train_time:.3f}s, Add: {add_time:.3f}s")
        
        start = time.time()
        ivf_results = [ivf.search(q, k=k) for q in queries]
        search_times["ivf"] = time.time() - start
        ivf_qps = n_queries / search_times["ivf"]
        print(f"  Search: {ivf_qps:.0f} QPS ({search_times['ivf']/n_queries*1000:.2f}ms/query)")
        
        results["ivf"] = ivf_results
        
        # =====================================================
        # Compute recalls
        # =====================================================
        print("\n[Recall Comparison]")
        
        for index_name in ["hnsw", "ivf"]:
            recalls = []
            for flat_res, other_res in zip(results["flat"], results[index_name]):
                flat_ids = {r.id for r in flat_res}
                other_ids = {r.id for r in other_res}
                recall = len(flat_ids & other_ids) / k
                recalls.append(recall)
            
            avg_recall = np.mean(recalls)
            print(f"  {index_name.upper()}: Recall@{k} = {avg_recall:.1%}")
        
        # =====================================================
        # Summary
        # =====================================================
        print("\n[Summary]")
        print(f"  {'Index':<10} | {'Build':>10} | {'QPS':>10} | {'Speedup':>10}")
        print("  " + "-" * 50)
        
        flat_qps = n_queries / search_times["flat"]
        
        for name in ["flat", "hnsw", "ivf"]:
            qps = n_queries / search_times[name]
            speedup = qps / flat_qps
            print(f"  {name:<10} | {build_times[name]:>9.2f}s | {qps:>10.0f} | {speedup:>9.1f}x")
        
        # Memory
        print("\n  Memory Usage:")
        print(f"    Flat: {flat.stats().memory_bytes / 1024 / 1024:.1f} MB")
        print(f"    HNSW: {hnsw.stats().memory_bytes / 1024 / 1024:.1f} MB")
        print(f"    IVF:  {ivf.stats().memory_bytes / 1024 / 1024:.1f} MB")
    
    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()