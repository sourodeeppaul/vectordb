"""
Benchmarks for VectorDB indexing operations.

Measures index building performance across different:
- Index types (Flat, IVF, HNSW, PQ)
- Dataset sizes
- Vector dimensions
- Batch sizes

Usage:
    python -m tests.benchmark.bench_indexing
    python -m tests.benchmark.bench_indexing --size large --index hnsw
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
    get_benchmark_config,
    SMALL_DATASET,
    MEDIUM_DATASET,
    LARGE_DATASET,
)

from vectordb.core.database import VectorDatabase
from vectordb.index.flat import FlatIndex
from vectordb.index.ivf import IVFIndex
from vectordb.index.hnsw import HNSWIndex
from vectordb.index.pq import ProductQuantizationIndex


@dataclass
class IndexBenchmarkConfig:
    """Configuration for index benchmarks."""
    
    n_vectors: int = 10000
    dimension: int = 128
    batch_sizes: List[int] = None
    index_types: List[str] = None
    warmup_runs: int = 1
    benchmark_runs: int = 5
    output_dir: str = "benchmark_results"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 10, 100, 1000]
        if self.index_types is None:
            self.index_types = ["flat", "ivf", "hnsw", "pq"]


class IndexingBenchmark:
    """Benchmark suite for indexing operations."""
    
    def __init__(self, config: IndexBenchmarkConfig):
        self.config = config
        self.runner = BenchmarkRunner(
            warmup_runs=config.warmup_runs,
            benchmark_runs=config.benchmark_runs
        )
        self.results: List[Dict[str, Any]] = []
        
        # Generate test data once
        print(f"Generating {config.n_vectors} vectors of dimension {config.dimension}...")
        self.vectors = generate_test_data(config.n_vectors, config.dimension)
        self.ids = [f"vec_{i}" for i in range(config.n_vectors)]
        
        # Temp directory for database
        self.temp_dir = tempfile.mkdtemp(prefix="vectordb_bench_")
    
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_index(self, index_type: str, dimension: int) -> Any:
        """Create an index of the specified type."""
        if index_type == "flat":
            return FlatIndex(dimension=dimension, metric="euclidean")
        elif index_type == "ivf":
            n_clusters = min(100, self.config.n_vectors // 100)
            return IVFIndex(
                dimension=dimension,
                n_clusters=n_clusters,
                metric="euclidean"
            )
        elif index_type == "hnsw":
            return HNSWIndex(
                dimension=dimension,
                M=16,
                ef_construction=100,
                metric="euclidean"
            )
        elif index_type == "pq":
            n_subvectors = min(16, dimension // 4)
            return ProductQuantizationIndex(
                dimension=dimension,
                n_subvectors=n_subvectors,
                n_bits=8,
                metric="euclidean"
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def benchmark_index_creation(self) -> List[BenchmarkResult]:
        """Benchmark index creation (empty index initialization)."""
        results = []
        
        print("\n=== Index Creation Benchmark ===")
        
        for index_type in self.config.index_types:
            def create_index():
                return self._create_index(index_type, self.config.dimension)
            
            result = self.runner.run_benchmark(
                name=f"{index_type}_creation",
                operation="create",
                func=create_index,
                dataset_size=0,
                dimension=self.config.dimension,
                index_type=index_type
            )
            
            print(f"  {index_type}: {result.mean_time*1000:.3f} ms")
            results.append(result)
        
        return results
    
    def benchmark_single_insert(self) -> List[BenchmarkResult]:
        """Benchmark single vector insertion."""
        results = []
        
        print("\n=== Single Insert Benchmark ===")
        
        for index_type in self.config.index_types:
            # Create fresh index for each test
            index = self._create_index(index_type, self.config.dimension)
            
            # For IVF, we need to train first
            if index_type == "ivf":
                train_data = self.vectors[:1000]
                index.train(train_data)
            
            insert_count = 0
            
            def insert_single():
                nonlocal insert_count
                idx = insert_count % self.config.n_vectors
                index.add(
                    self.vectors[idx:idx+1],
                    ids=[self.ids[idx]]
                )
                insert_count += 1
            
            result = self.runner.run_benchmark(
                name=f"{index_type}_single_insert",
                operation="insert",
                func=insert_single,
                dataset_size=1,
                dimension=self.config.dimension,
                index_type=index_type
            )
            
            print(f"  {index_type}: {result.mean_time*1000:.3f} ms ({result.operations_per_second:.0f} ops/sec)")
            results.append(result)
        
        return results
    
    def benchmark_batch_insert(self) -> List[BenchmarkResult]:
        """Benchmark batch vector insertion with different batch sizes."""
        results = []
        
        print("\n=== Batch Insert Benchmark ===")
        
        for index_type in self.config.index_types:
            print(f"\n  {index_type}:")
            
            for batch_size in self.config.batch_sizes:
                if batch_size > self.config.n_vectors:
                    continue
                
                # Create fresh index for each batch size test
                index = self._create_index(index_type, self.config.dimension)
                
                # Train IVF if needed
                if index_type == "ivf":
                    index.train(self.vectors[:min(1000, self.config.n_vectors)])
                
                batch_idx = 0
                
                def insert_batch():
                    nonlocal batch_idx
                    start = (batch_idx * batch_size) % (self.config.n_vectors - batch_size)
                    end = start + batch_size
                    index.add(
                        self.vectors[start:end],
                        ids=self.ids[start:end]
                    )
                    batch_idx += 1
                
                result = self.runner.run_batch_benchmark(
                    name=f"{index_type}_batch_{batch_size}",
                    operation="batch_insert",
                    func=insert_batch,
                    batch_size=batch_size,
                    dataset_size=batch_size,
                    dimension=self.config.dimension,
                    index_type=index_type
                )
                
                print(f"    batch_size={batch_size}: {result.operations_per_second:.0f} vectors/sec")
                results.append(result)
        
        return results
    
    def benchmark_full_build(self) -> List[BenchmarkResult]:
        """Benchmark building a complete index from scratch."""
        results = []
        
        print("\n=== Full Index Build Benchmark ===")
        
        for index_type in self.config.index_types:
            times = []
            
            for _ in range(self.config.benchmark_runs):
                index = self._create_index(index_type, self.config.dimension)
                
                start = time.perf_counter()
                
                # Train if needed
                if index_type in ["ivf", "pq"]:
                    index.train(self.vectors)
                
                # Add all vectors
                index.add(self.vectors, ids=self.ids)
                
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            times = np.array(times)
            
            result = BenchmarkResult(
                name=f"{index_type}_full_build",
                operation="full_build",
                dataset_size=self.config.n_vectors,
                dimension=self.config.dimension,
                total_time=np.sum(times),
                mean_time=np.mean(times),
                std_time=np.std(times),
                min_time=np.min(times),
                max_time=np.max(times),
                operations_per_second=self.config.n_vectors / np.mean(times),
                extra_metrics={"index_type": index_type}
            )
            
            print(f"  {index_type}: {result.mean_time:.3f}s ({result.operations_per_second:.0f} vectors/sec)")
            results.append(result)
            self.runner.results.append(result)
        
        return results
    
    def benchmark_incremental_build(self) -> List[BenchmarkResult]:
        """Benchmark incremental index building (add vectors in chunks)."""
        results = []
        chunk_size = self.config.n_vectors // 10
        
        print(f"\n=== Incremental Build Benchmark (chunk_size={chunk_size}) ===")
        
        for index_type in self.config.index_types:
            times = []
            
            for _ in range(self.config.benchmark_runs):
                index = self._create_index(index_type, self.config.dimension)
                
                # Initial training for IVF/PQ
                if index_type in ["ivf", "pq"]:
                    index.train(self.vectors[:chunk_size])
                
                chunk_times = []
                for i in range(0, self.config.n_vectors, chunk_size):
                    start = time.perf_counter()
                    end = min(i + chunk_size, self.config.n_vectors)
                    index.add(self.vectors[i:end], ids=self.ids[i:end])
                    chunk_times.append(time.perf_counter() - start)
                
                times.append(sum(chunk_times))
            
            times = np.array(times)
            
            result = BenchmarkResult(
                name=f"{index_type}_incremental_build",
                operation="incremental_build",
                dataset_size=self.config.n_vectors,
                dimension=self.config.dimension,
                total_time=np.sum(times),
                mean_time=np.mean(times),
                std_time=np.std(times),
                min_time=np.min(times),
                max_time=np.max(times),
                operations_per_second=self.config.n_vectors / np.mean(times),
                extra_metrics={
                    "index_type": index_type,
                    "chunk_size": chunk_size
                }
            )
            
            print(f"  {index_type}: {result.mean_time:.3f}s")
            results.append(result)
            self.runner.results.append(result)
        
        return results
    
    def benchmark_database_operations(self) -> List[BenchmarkResult]:
        """Benchmark full database operations including persistence."""
        results = []
        
        print("\n=== Database Operations Benchmark ===")
        
        for index_type in self.config.index_types:
            db_path = Path(self.temp_dir) / f"db_{index_type}"
            
            times = []
            
            for run in range(self.config.benchmark_runs):
                # Clean up previous run
                if db_path.exists():
                    shutil.rmtree(db_path)
                
                start = time.perf_counter()
                
                # Create database and collection
                db = VectorDatabase(storage_path=str(db_path))
                collection = db.create_collection(
                    name="benchmark",
                    dimension=self.config.dimension,
                    index_type=index_type
                )
                
                # Add vectors in batches
                batch_size = 1000
                for i in range(0, self.config.n_vectors, batch_size):
                    end = min(i + batch_size, self.config.n_vectors)
                    collection.add(
                        self.vectors[i:end],
                        ids=self.ids[i:end]
                    )
                
                # Close (triggers persistence)
                db.close()
                
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            times = np.array(times)
            
            result = BenchmarkResult(
                name=f"{index_type}_database_build",
                operation="database_build",
                dataset_size=self.config.n_vectors,
                dimension=self.config.dimension,
                total_time=np.sum(times),
                mean_time=np.mean(times),
                std_time=np.std(times),
                min_time=np.min(times),
                max_time=np.max(times),
                operations_per_second=self.config.n_vectors / np.mean(times),
                extra_metrics={"index_type": index_type, "includes_persistence": True}
            )
            
            print(f"  {index_type}: {result.mean_time:.3f}s (with persistence)")
            results.append(result)
            self.runner.results.append(result)
        
        return results
    
    def benchmark_dimension_scaling(self) -> List[BenchmarkResult]:
        """Benchmark how indexing scales with vector dimension."""
        results = []
        dimensions = [32, 64, 128, 256, 512, 768, 1024]
        n_vectors = 5000  # Fixed smaller size for dimension tests
        
        print("\n=== Dimension Scaling Benchmark ===")
        
        for dim in dimensions:
            vectors = generate_test_data(n_vectors, dim)
            ids = [f"vec_{i}" for i in range(n_vectors)]
            
            # Test with HNSW as representative ANN index
            times = []
            
            for _ in range(self.config.benchmark_runs):
                index = HNSWIndex(dimension=dim, M=16, ef_construction=100)
                
                start = time.perf_counter()
                index.add(vectors, ids=ids)
                elapsed = time.perf_counter() - start
                
                times.append(elapsed)
            
            times = np.array(times)
            
            result = BenchmarkResult(
                name=f"hnsw_dim_{dim}",
                operation="dimension_scaling",
                dataset_size=n_vectors,
                dimension=dim,
                total_time=np.sum(times),
                mean_time=np.mean(times),
                std_time=np.std(times),
                min_time=np.min(times),
                max_time=np.max(times),
                operations_per_second=n_vectors / np.mean(times),
                extra_metrics={"index_type": "hnsw"}
            )
            
            print(f"  dim={dim}: {result.mean_time:.3f}s ({result.operations_per_second:.0f} vectors/sec)")
            results.append(result)
            self.runner.results.append(result)
        
        return results
    
    def benchmark_size_scaling(self) -> List[BenchmarkResult]:
        """Benchmark how indexing scales with dataset size."""
        results = []
        sizes = [1000, 5000, 10000, 25000, 50000]
        
        print("\n=== Size Scaling Benchmark ===")
        
        for size in sizes:
            if size > self.config.n_vectors:
                # Generate more data if needed
                vectors = generate_test_data(size, self.config.dimension)
                ids = [f"vec_{i}" for i in range(size)]
            else:
                vectors = self.vectors[:size]
                ids = self.ids[:size]
            
            # Test with HNSW
            times = []
            
            for _ in range(max(1, self.config.benchmark_runs // 2)):
                index = HNSWIndex(
                    dimension=self.config.dimension,
                    M=16,
                    ef_construction=100
                )
                
                start = time.perf_counter()
                index.add(vectors, ids=ids)
                elapsed = time.perf_counter() - start
                
                times.append(elapsed)
            
            times = np.array(times)
            
            result = BenchmarkResult(
                name=f"hnsw_size_{size}",
                operation="size_scaling",
                dataset_size=size,
                dimension=self.config.dimension,
                total_time=np.sum(times),
                mean_time=np.mean(times),
                std_time=np.std(times),
                min_time=np.min(times),
                max_time=np.max(times),
                operations_per_second=size / np.mean(times),
                extra_metrics={"index_type": "hnsw"}
            )
            
            print(f"  n={size}: {result.mean_time:.3f}s ({result.operations_per_second:.0f} vectors/sec)")
            results.append(result)
            self.runner.results.append(result)
        
        return results
    
    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all indexing benchmarks."""
        all_results = {}
        
        try:
            all_results["creation"] = self.benchmark_index_creation()
            all_results["single_insert"] = self.benchmark_single_insert()
            all_results["batch_insert"] = self.benchmark_batch_insert()
            all_results["full_build"] = self.benchmark_full_build()
            all_results["incremental_build"] = self.benchmark_incremental_build()
            all_results["database_operations"] = self.benchmark_database_operations()
            all_results["dimension_scaling"] = self.benchmark_dimension_scaling()
            all_results["size_scaling"] = self.benchmark_size_scaling()
            
            return all_results
        finally:
            self.cleanup()
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        self.runner.save_results(output_path)
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VectorDB Indexing Benchmarks")
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
        "--dimension",
        type=int,
        default=128,
        help="Vector dimension"
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Specific index type to benchmark (default: all)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/indexing.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    if args.n_vectors:
        n_vectors = args.n_vectors
    else:
        preset = get_benchmark_config(args.size)
        n_vectors = preset["n_vectors"]
    
    index_types = [args.index] if args.index else None
    
    config = IndexBenchmarkConfig(
        n_vectors=n_vectors,
        dimension=args.dimension,
        index_types=index_types,
        benchmark_runs=args.runs
    )
    
    print("=" * 60)
    print("VectorDB Indexing Benchmark")
    print("=" * 60)
    print(f"Vectors: {config.n_vectors}")
    print(f"Dimension: {config.dimension}")
    print(f"Index types: {config.index_types}")
    print(f"Benchmark runs: {config.benchmark_runs}")
    
    benchmark = IndexingBenchmark(config)
    benchmark.run_all()
    benchmark.save_results(args.output)
    benchmark.runner.print_summary()


if __name__ == "__main__":
    main()