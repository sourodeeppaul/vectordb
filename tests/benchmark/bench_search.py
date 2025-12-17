"""
Benchmarks for VectorDB search operations.

Measures search performance across different:
- Index types
- Query batch sizes
- k values (number of neighbors)
- Filter complexities

Usage:
    python -m tests.benchmark.bench_search
    python -m tests.benchmark.bench_search --size large --k 100
"""

import argparse
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from tests.benchmark import (
    BenchmarkRunner,
    BenchmarkResult,
    Timer,
    generate_test_data,
    generate_query_data,
    get_benchmark_config,
)

from vectordb.core.database import VectorDatabase
from vectordb.index.flat import FlatIndex
from vectordb.index.ivf import IVFIndex
from vectordb.index.hnsw import HNSWIndex
from vectordb.index.pq import ProductQuantizationIndex


@dataclass
class SearchBenchmarkConfig:
    """Configuration for search benchmarks."""
    
    n_vectors: int = 10000
    n_queries: int = 100
    dimension: int = 128
    k_values: List[int] = None
    index_types: List[str] = None
    warmup_runs: int = 3
    benchmark_runs: int = 10
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 10, 50, 100]
        if self.index_types is None:
            self.index_types = ["flat", "ivf", "hnsw"]


class SearchBenchmark:
    """Benchmark suite for search operations."""
    
    def __init__(self, config: SearchBenchmarkConfig):
        self.config = config
        self.runner = BenchmarkRunner(
            warmup_runs=config.warmup_runs,
            benchmark_runs=config.benchmark_runs
        )
        
        # Generate test data
        print(f"Generating {config.n_vectors} vectors...")
        self.vectors = generate_test_data(config.n_vectors, config.dimension)
        self.ids = [f"vec_{i}" for i in range(config.n_vectors)]
        
        print(f"Generating {config.n_queries} queries...")
        self.queries = generate_query_data(config.n_queries, config.dimension)
        
        # Generate metadata for filter tests
        self.metadata = self._generate_metadata()
        
        # Pre-build indexes
        self.indexes = {}
        self._build_indexes()
        
        # Temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="vectordb_search_bench_")
    
    def _generate_metadata(self) -> List[Dict[str, Any]]:
        """Generate metadata for filter benchmarks."""
        categories = ["A", "B", "C", "D", "E"]
        return [
            {
                "category": categories[i % len(categories)],
                "value": float(i % 100),
                "active": i % 2 == 0,
                "score": np.random.rand(),
            }
            for i in range(self.config.n_vectors)
        ]
    
    def _build_indexes(self):
        """Pre-build all indexes for search benchmarks."""
        print("\nBuilding indexes...")
        
        for index_type in self.config.index_types:
            print(f"  Building {index_type}...", end=" ", flush=True)
            start = time.perf_counter()
            
            if index_type == "flat":
                index = FlatIndex(dimension=self.config.dimension)
            elif index_type == "ivf":
                n_clusters = min(100, self.config.n_vectors // 100)
                index = IVFIndex(
                    dimension=self.config.dimension,
                    n_clusters=n_clusters
                )
                index.train(self.vectors)
            elif index_type == "hnsw":
                index = HNSWIndex(
                    dimension=self.config.dimension,
                    M=16,
                    ef_construction=100
                )
            elif index_type == "pq":
                index = ProductQuantizationIndex(
                    dimension=self.config.dimension,
                    n_subvectors=16,
                    n_bits=8
                )
                index.train(self.vectors)
            else:
                continue
            
            index.add(self.vectors, ids=self.ids, metadata=self.metadata)
            self.indexes[index_type] = index
            
            elapsed = time.perf_counter() - start
            print(f"{elapsed:.2f}s")
    
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def benchmark_single_query(self) -> List[BenchmarkResult]:
        """Benchmark single query latency."""
        results = []
        k = 10
        
        print("\n=== Single Query Latency (k=10) ===")
        
        for index_type in self.config.index_types:
            if index_type not in self.indexes:
                continue
            
            index = self.indexes[index_type]
            query_idx = 0
            
            def single_query():
                nonlocal query_idx
                query = self.queries[query_idx % self.config.n_queries]
                index.search(query, k=k)
                query_idx += 1
            
            result = self.runner.run_benchmark(
                name=f"{index_type}_single_query",
                operation="single_query",
                func=single_query,
                dataset_size=self.config.n_vectors,
                dimension=self.config.dimension,
                index_type=index_type,
                k=k
            )
            
            latency_ms = result.mean_time * 1000
            qps = result.operations_per_second
            print(f"  {index_type}: {latency_ms:.3f} ms ({qps:.0f} QPS)")
            results.append(result)
        
        return results
    
    def benchmark_varying_k(self) -> List[BenchmarkResult]:
        """Benchmark search with different k values."""
        results = []
        
        print("\n=== Varying k Benchmark ===")
        
        for index_type in self.config.index_types:
            if index_type not in self.indexes:
                continue
            
            print(f"\n  {index_type}:")
            index = self.indexes[index_type]
            
            for k in self.config.k_values:
                query_idx = 0
                
                def search_k():
                    nonlocal query_idx
                    query = self.queries[query_idx % self.config.n_queries]
                    index.search(query, k=k)
                    query_idx += 1
                
                result = self.runner.run_benchmark(
                    name=f"{index_type}_k{k}",
                    operation=f"search_k{k}",
                    func=search_k,
                    dataset_size=self.config.n_vectors,
                    dimension=self.config.dimension,
                    index_type=index_type,
                    k=k
                )
                
                print(f"    k={k}: {result.mean_time*1000:.3f} ms")
                results.append(result)
        
        return results
    
    def benchmark_batch_query(self) -> List[BenchmarkResult]:
        """Benchmark batch query performance."""
        results = []
        batch_sizes = [1, 10, 50, 100]
        k = 10
        
        print("\n=== Batch Query Benchmark (k=10) ===")
        
        for index_type in self.config.index_types:
            if index_type not in self.indexes:
                continue
            
            print(f"\n  {index_type}:")
            index = self.indexes[index_type]
            
            for batch_size in batch_sizes:
                batch_queries = self.queries[:batch_size]
                
                def batch_search():
                    index.search_batch(batch_queries, k=k)
                
                result = self.runner.run_batch_benchmark(
                    name=f"{index_type}_batch{batch_size}",
                    operation="batch_query",
                    func=batch_search,
                    batch_size=batch_size,
                    dataset_size=self.config.n_vectors,
                    dimension=self.config.dimension,
                    index_type=index_type,
                    k=k
                )
                
                total_time_ms = result.mean_time * batch_size * 1000
                per_query_ms = result.mean_time * 1000
                print(f"    batch={batch_size}: {total_time_ms:.3f} ms total, {per_query_ms:.3f} ms/query")
                results.append(result)
        
        return results
    
    def benchmark_filtered_search(self) -> List[BenchmarkResult]:
        """Benchmark search with metadata filters."""
        results = []
        k = 10
        
        print("\n=== Filtered Search Benchmark ===")
        
        # Different filter complexities
        filters = {
            "simple_eq": {"category": "A"},
            "simple_bool": {"active": True},
            "range": {"value": {"$gte": 25, "$lt": 75}},
            "compound": {
                "$and": [
                    {"category": {"$in": ["A", "B"]}},
                    {"active": True}
                ]
            }
        }
        
        for index_type in self.config.index_types:
            if index_type not in self.indexes:
                continue
            
            print(f"\n  {index_type}:")
            index = self.indexes[index_type]
            
            # First measure unfiltered baseline
            def unfiltered_search():
                query = self.queries[0]
                index.search(query, k=k)
            
            baseline = self.runner.run_benchmark(
                name=f"{index_type}_unfiltered",
                operation="unfiltered_search",
                func=unfiltered_search,
                dataset_size=self.config.n_vectors,
                dimension=self.config.dimension,
                index_type=index_type
            )
            print(f"    unfiltered: {baseline.mean_time*1000:.3f} ms")
            results.append(baseline)
            
            # Benchmark each filter type
            for filter_name, filter_config in filters.items():
                def filtered_search():
                    query = self.queries[0]
                    index.search(query, k=k, filter=filter_config)
                
                result = self.runner.run_benchmark(
                    name=f"{index_type}_{filter_name}",
                    operation=f"filtered_{filter_name}",
                    func=filtered_search,
                    dataset_size=self.config.n_vectors,
                    dimension=self.config.dimension,
                    index_type=index_type,
                    filter_type=filter_name
                )
                
                overhead = (result.mean_time / baseline.mean_time - 1) * 100
                print(f"    {filter_name}: {result.mean_time*1000:.3f} ms (+{overhead:.1f}%)")
                results.append(result)
        
        return results
    
    def benchmark_concurrent_queries(self) -> List[BenchmarkResult]:
        """Benchmark concurrent query execution."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        k = 10
        n_concurrent = [1, 2, 4, 8]
        
        print("\n=== Concurrent Query Benchmark ===")
        
        for index_type in self.config.index_types:
            if index_type not in self.indexes:
                continue
            
            print(f"\n  {index_type}:")
            index = self.indexes[index_type]
            
            for n_threads in n_concurrent:
                def run_concurrent():
                    with ThreadPoolExecutor(max_workers=n_threads) as executor:
                        futures = []
                        for i in range(n_threads * 10):
                            query = self.queries[i % self.config.n_queries]
                            futures.append(executor.submit(index.search, query, k))
                        for f in as_completed(futures):
                            f.result()
                
                times = []
                for _ in range(self.config.benchmark_runs):
                    start = time.perf_counter()
                    run_concurrent()
                    times.append(time.perf_counter() - start)
                
                times = np.array(times)
                n_queries = n_threads * 10
                
                result = BenchmarkResult(
                    name=f"{index_type}_concurrent_{n_threads}",
                    operation="concurrent_query",
                    dataset_size=self.config.n_vectors,
                    dimension=self.config.dimension,
                    total_time=np.sum(times),
                    mean_time=np.mean(times) / n_queries,
                    std_time=np.std(times) / n_queries,
                    min_time=np.min(times) / n_queries,
                    max_time=np.max(times) / n_queries,
                    operations_per_second=n_queries / np.mean(times),
                    extra_metrics={
                        "index_type": index_type,
                        "n_threads": n_threads,
                        "k": k
                    }
                )
                
                throughput = result.operations_per_second
                print(f"    threads={n_threads}: {throughput:.0f} QPS")
                results.append(result)
                self.runner.results.append(result)
        
        return results
    
    def benchmark_hnsw_ef_search(self) -> List[BenchmarkResult]:
        """Benchmark HNSW with different ef_search values."""
        results = []
        k = 10
        ef_values = [16, 32, 64, 128, 256]
        
        print("\n=== HNSW ef_search Benchmark ===")
        
        if "hnsw" not in self.indexes:
            print("  HNSW index not available")
            return results
        
        index = self.indexes["hnsw"]
        
        for ef in ef_values:
            # Set ef_search parameter
            index.ef_search = ef
            
            query_idx = 0
            
            def search_with_ef():
                nonlocal query_idx
                query = self.queries[query_idx % self.config.n_queries]
                index.search(query, k=k)
                query_idx += 1
            
            result = self.runner.run_benchmark(
                name=f"hnsw_ef{ef}",
                operation="hnsw_ef_search",
                func=search_with_ef,
                dataset_size=self.config.n_vectors,
                dimension=self.config.dimension,
                index_type="hnsw",
                ef_search=ef,
                k=k
            )
            
            print(f"  ef={ef}: {result.mean_time*1000:.3f} ms ({result.operations_per_second:.0f} QPS)")
            results.append(result)
        
        return results
    
    def benchmark_ivf_nprobe(self) -> List[BenchmarkResult]:
        """Benchmark IVF with different nprobe values."""
        results = []
        k = 10
        nprobe_values = [1, 2, 4, 8, 16, 32]
        
        print("\n=== IVF nprobe Benchmark ===")
        
        if "ivf" not in self.indexes:
            print("  IVF index not available")
            return results
        
        index = self.indexes["ivf"]
        
        for nprobe in nprobe_values:
            # Set nprobe parameter
            index.nprobe = nprobe
            
            query_idx = 0
            
            def search_with_nprobe():
                nonlocal query_idx
                query = self.queries[query_idx % self.config.n_queries]
                index.search(query, k=k)
                query_idx += 1
            
            result = self.runner.run_benchmark(
                name=f"ivf_nprobe{nprobe}",
                operation="ivf_nprobe_search",
                func=search_with_nprobe,
                dataset_size=self.config.n_vectors,
                dimension=self.config.dimension,
                index_type="ivf",
                nprobe=nprobe,
                k=k
            )
            
            print(f"  nprobe={nprobe}: {result.mean_time*1000:.3f} ms ({result.operations_per_second:.0f} QPS)")
            results.append(result)
        
        return results
    
    def benchmark_with_database(self) -> List[BenchmarkResult]:
        """Benchmark search through full database stack."""
        results = []
        k = 10
        
        print("\n=== Full Database Stack Benchmark ===")
        
        for index_type in self.config.index_types:
            db_path = Path(self.temp_dir) / f"db_{index_type}"
            
            # Create database with data
            db = VectorDatabase(storage_path=str(db_path))
            collection = db.create_collection(
                name="benchmark",
                dimension=self.config.dimension,
                index_type=index_type
            )
            collection.add(self.vectors, ids=self.ids, metadata=self.metadata)
            
            query_idx = 0
            
            def db_search():
                nonlocal query_idx
                query = self.queries[query_idx % self.config.n_queries]
                collection.search(query, k=k)
                query_idx += 1
            
            result = self.runner.run_benchmark(
                name=f"{index_type}_database_search",
                operation="database_search",
                func=db_search,
                dataset_size=self.config.n_vectors,
                dimension=self.config.dimension,
                index_type=index_type,
                k=k
            )
            
            print(f"  {index_type}: {result.mean_time*1000:.3f} ms ({result.operations_per_second:.0f} QPS)")
            results.append(result)
            
            db.close()
        
        return results
    
    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all search benchmarks."""
        all_results = {}
        
        try:
            all_results["single_query"] = self.benchmark_single_query()
            all_results["varying_k"] = self.benchmark_varying_k()
            all_results["batch_query"] = self.benchmark_batch_query()
            all_results["filtered_search"] = self.benchmark_filtered_search()
            all_results["concurrent_queries"] = self.benchmark_concurrent_queries()
            all_results["hnsw_ef_search"] = self.benchmark_hnsw_ef_search()
            all_results["ivf_nprobe"] = self.benchmark_ivf_nprobe()
            all_results["database_search"] = self.benchmark_with_database()
            
            return all_results
        finally:
            self.cleanup()
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        self.runner.save_results(output_path)
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VectorDB Search Benchmarks")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Dataset size preset"
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=None,
        help="Number of vectors (overrides --size)"
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Number of query vectors"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=128,
        help="Vector dimension"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="k values to benchmark"
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Specific index type to benchmark"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/search.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    if args.n_vectors:
        n_vectors = args.n_vectors
    else:
        preset = get_benchmark_config(args.size)
        n_vectors = preset["n_vectors"]
    
    config = SearchBenchmarkConfig(
        n_vectors=n_vectors,
        n_queries=args.n_queries,
        dimension=args.dimension,
        k_values=args.k,
        index_types=[args.index] if args.index else None,
        benchmark_runs=args.runs
    )
    
    print("=" * 60)
    print("VectorDB Search Benchmark")
    print("=" * 60)
    print(f"Dataset: {config.n_vectors} vectors x {config.dimension} dims")
    print(f"Queries: {config.n_queries}")
    print(f"k values: {config.k_values}")
    print(f"Index types: {config.index_types}")
    
    benchmark = SearchBenchmark(config)
    benchmark.run_all()
    benchmark.save_results(args.output)
    benchmark.runner.print_summary()


if __name__ == "__main__":
    main()