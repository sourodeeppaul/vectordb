"""
Benchmarks for VectorDB recall accuracy.

Measures recall@k for approximate nearest neighbor indexes
compared to exact (brute-force) search.

Usage:
    python -m tests.benchmark.bench_recall
    python -m tests.benchmark.bench_recall --index hnsw --k 100
"""

import argparse
import numpy as np
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from tests.benchmark import (
    RecallResult,
    generate_test_data,
    generate_query_data,
    compute_ground_truth,
    compute_recall,
    get_benchmark_config,
)

from vectordb.index.flat import FlatIndex
from vectordb.index.ivf import IVFIndex
from vectordb.index.hnsw import HNSWIndex
from vectordb.index.pq import ProductQuantizationIndex


@dataclass
class RecallBenchmarkConfig:
    """Configuration for recall benchmarks."""
    
    n_vectors: int = 10000
    n_queries: int = 100
    dimension: int = 128
    k_values: List[int] = None
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 10, 50, 100]
        if self.metrics is None:
            self.metrics = ["euclidean", "cosine"]


class RecallBenchmark:
    """Benchmark suite for recall accuracy."""
    
    def __init__(self, config: RecallBenchmarkConfig):
        self.config = config
        self.results: List[RecallResult] = []
        
        # Generate test data
        print(f"Generating {config.n_vectors} vectors...")
        self.vectors = generate_test_data(config.n_vectors, config.dimension)
        self.ids = [f"vec_{i}" for i in range(config.n_vectors)]
        
        print(f"Generating {config.n_queries} queries...")
        self.queries = generate_query_data(config.n_queries, config.dimension)
        
        # Compute ground truth for each metric
        self.ground_truth = {}
        for metric in config.metrics:
            print(f"Computing ground truth for {metric}...")
            max_k = max(config.k_values)
            self.ground_truth[metric] = compute_ground_truth(
                self.vectors, self.queries, max_k, metric
            )
    
    def _get_results_as_indices(
        self,
        index: Any,
        k: int
    ) -> List[List[int]]:
        """Convert search results to list of indices."""
        all_indices = []
        
        for query in self.queries:
            results = index.search(query, k=k)
            # Extract indices from results
            indices = []
            for r in results:
                # Find index from ID
                vec_id = r.get("id", r.get("idx", None))
                if vec_id is not None:
                    if isinstance(vec_id, str) and vec_id.startswith("vec_"):
                        idx = int(vec_id.split("_")[1])
                    elif isinstance(vec_id, int):
                        idx = vec_id
                    else:
                        idx = self.ids.index(vec_id)
                    indices.append(idx)
            all_indices.append(indices)
        
        return all_indices
    
    def benchmark_flat_baseline(self, metric: str) -> Dict[str, float]:
        """Establish baseline with flat (exact) index."""
        print(f"\n  Computing Flat index baseline ({metric})...")
        
        index = FlatIndex(dimension=self.config.dimension, metric=metric)
        
        start = time.perf_counter()
        index.add(self.vectors, ids=self.ids)
        build_time = time.perf_counter() - start
        
        # Verify recall is 100%
        max_k = max(self.config.k_values)
        predicted = self._get_results_as_indices(index, max_k)
        recall = compute_recall(predicted, self.ground_truth[metric], max_k)
        
        # Measure search time
        search_times = []
        for query in self.queries:
            start = time.perf_counter()
            index.search(query, k=max_k)
            search_times.append(time.perf_counter() - start)
        
        return {
            "build_time": build_time,
            "mean_search_time": np.mean(search_times),
            "recall": recall
        }
    
    def benchmark_hnsw_recall(self, metric: str = "euclidean") -> List[RecallResult]:
        """Benchmark HNSW recall with different parameters."""
        results = []
        
        print(f"\n=== HNSW Recall ({metric}) ===")
        
        # Parameter configurations to test
        configs = [
            {"M": 8, "ef_construction": 50, "ef_search": 16},
            {"M": 8, "ef_construction": 50, "ef_search": 64},
            {"M": 16, "ef_construction": 100, "ef_search": 32},
            {"M": 16, "ef_construction": 100, "ef_search": 64},
            {"M": 16, "ef_construction": 100, "ef_search": 128},
            {"M": 32, "ef_construction": 200, "ef_search": 64},
            {"M": 32, "ef_construction": 200, "ef_search": 128},
            {"M": 32, "ef_construction": 200, "ef_search": 256},
        ]
        
        for params in configs:
            print(f"\n  M={params['M']}, ef_c={params['ef_construction']}, ef_s={params['ef_search']}")
            
            # Build index
            index = HNSWIndex(
                dimension=self.config.dimension,
                M=params["M"],
                ef_construction=params["ef_construction"],
                metric=metric
            )
            
            start = time.perf_counter()
            index.add(self.vectors, ids=self.ids)
            build_time = time.perf_counter() - start
            
            # Set search parameter
            index.ef_search = params["ef_search"]
            
            # Test each k value
            for k in self.config.k_values:
                # Get predictions
                predicted = self._get_results_as_indices(index, k)
                
                # Compute recall
                recall_at_k = compute_recall(predicted, self.ground_truth[metric], k)
                recall_at_1 = compute_recall(predicted, self.ground_truth[metric], 1)
                
                # Measure search time
                search_times = []
                for query in self.queries:
                    start = time.perf_counter()
                    index.search(query, k=k)
                    search_times.append(time.perf_counter() - start)
                
                result = RecallResult(
                    name=f"hnsw_M{params['M']}_ef{params['ef_search']}_k{k}",
                    index_type="hnsw",
                    dataset_size=self.config.n_vectors,
                    dimension=self.config.dimension,
                    k=k,
                    recall_at_k=recall_at_k,
                    recall_at_1=recall_at_1,
                    build_time=build_time,
                    mean_search_time=np.mean(search_times),
                    index_params=params
                )
                
                print(f"    k={k}: recall@k={recall_at_k:.4f}, recall@1={recall_at_1:.4f}, "
                      f"time={np.mean(search_times)*1000:.2f}ms")
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def benchmark_ivf_recall(self, metric: str = "euclidean") -> List[RecallResult]:
        """Benchmark IVF recall with different parameters."""
        results = []
        
        print(f"\n=== IVF Recall ({metric}) ===")
        
        # Parameter configurations
        n_clusters_options = [16, 64, 256]
        nprobe_options = [1, 4, 16, 64]
        
        for n_clusters in n_clusters_options:
            if n_clusters > self.config.n_vectors // 10:
                continue
            
            print(f"\n  n_clusters={n_clusters}")
            
            # Build index
            index = IVFIndex(
                dimension=self.config.dimension,
                n_clusters=n_clusters,
                metric=metric
            )
            
            start = time.perf_counter()
            index.train(self.vectors)
            index.add(self.vectors, ids=self.ids)
            build_time = time.perf_counter() - start
            
            for nprobe in nprobe_options:
                if nprobe > n_clusters:
                    continue
                
                index.nprobe = nprobe
                
                for k in self.config.k_values:
                    # Get predictions
                    predicted = self._get_results_as_indices(index, k)
                    
                    # Compute recall
                    recall_at_k = compute_recall(predicted, self.ground_truth[metric], k)
                    recall_at_1 = compute_recall(predicted, self.ground_truth[metric], 1)
                    
                    # Measure search time
                    search_times = []
                    for query in self.queries:
                        start = time.perf_counter()
                        index.search(query, k=k)
                        search_times.append(time.perf_counter() - start)
                    
                    result = RecallResult(
                        name=f"ivf_c{n_clusters}_np{nprobe}_k{k}",
                        index_type="ivf",
                        dataset_size=self.config.n_vectors,
                        dimension=self.config.dimension,
                        k=k,
                        recall_at_k=recall_at_k,
                        recall_at_1=recall_at_1,
                        build_time=build_time,
                        mean_search_time=np.mean(search_times),
                        index_params={"n_clusters": n_clusters, "nprobe": nprobe}
                    )
                    
                    print(f"    nprobe={nprobe}, k={k}: recall@k={recall_at_k:.4f}, "
                          f"time={np.mean(search_times)*1000:.2f}ms")
                    
                    results.append(result)
                    self.results.append(result)
        
        return results
    
    def benchmark_pq_recall(self, metric: str = "euclidean") -> List[RecallResult]:
        """Benchmark Product Quantization recall."""
        results = []
        
        print(f"\n=== PQ Recall ({metric}) ===")
        
        # Parameter configurations
        configs = [
            {"n_subvectors": 8, "n_bits": 8},
            {"n_subvectors": 16, "n_bits": 8},
            {"n_subvectors": 32, "n_bits": 8},
            {"n_subvectors": 16, "n_bits": 4},
        ]
        
        for params in configs:
            if params["n_subvectors"] > self.config.dimension:
                continue
            if self.config.dimension % params["n_subvectors"] != 0:
                continue
            
            print(f"\n  n_subvectors={params['n_subvectors']}, n_bits={params['n_bits']}")
            
            # Build index
            index = ProductQuantizationIndex(
                dimension=self.config.dimension,
                n_subvectors=params["n_subvectors"],
                n_bits=params["n_bits"],
                metric=metric
            )
            
            start = time.perf_counter()
            index.train(self.vectors)
            index.add(self.vectors, ids=self.ids)
            build_time = time.perf_counter() - start
            
            for k in self.config.k_values:
                # Get predictions
                predicted = self._get_results_as_indices(index, k)
                
                # Compute recall
                recall_at_k = compute_recall(predicted, self.ground_truth[metric], k)
                recall_at_1 = compute_recall(predicted, self.ground_truth[metric], 1)
                
                # Measure search time
                search_times = []
                for query in self.queries:
                    start = time.perf_counter()
                    index.search(query, k=k)
                    search_times.append(time.perf_counter() - start)
                
                result = RecallResult(
                    name=f"pq_m{params['n_subvectors']}_b{params['n_bits']}_k{k}",
                    index_type="pq",
                    dataset_size=self.config.n_vectors,
                    dimension=self.config.dimension,
                    k=k,
                    recall_at_k=recall_at_k,
                    recall_at_1=recall_at_1,
                    build_time=build_time,
                    mean_search_time=np.mean(search_times),
                    index_params=params
                )
                
                print(f"    k={k}: recall@k={recall_at_k:.4f}, time={np.mean(search_times)*1000:.2f}ms")
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def benchmark_recall_vs_speed(self, metric: str = "euclidean") -> Dict[str, Any]:
        """Generate recall vs speed trade-off data."""
        print(f"\n=== Recall vs Speed Trade-off ({metric}) ===")
        
        tradeoff_data = []
        k = 10
        
        # Flat (baseline)
        baseline = self.benchmark_flat_baseline(metric)
        tradeoff_data.append({
            "index": "flat",
            "params": {},
            "recall": baseline["recall"],
            "qps": 1.0 / baseline["mean_search_time"],
            "build_time": baseline["build_time"]
        })
        
        # HNSW configurations
        for ef_search in [16, 32, 64, 128, 256]:
            index = HNSWIndex(
                dimension=self.config.dimension,
                M=16,
                ef_construction=100,
                metric=metric
            )
            index.add(self.vectors, ids=self.ids)
            index.ef_search = ef_search
            
            predicted = self._get_results_as_indices(index, k)
            recall = compute_recall(predicted, self.ground_truth[metric], k)
            
            search_times = [
                self._time_search(index, q, k) for q in self.queries[:20]
            ]
            
            tradeoff_data.append({
                "index": "hnsw",
                "params": {"M": 16, "ef_search": ef_search},
                "recall": recall,
                "qps": 1.0 / np.mean(search_times),
                "build_time": 0
            })
        
        # IVF configurations
        for nprobe in [1, 2, 4, 8, 16, 32]:
            index = IVFIndex(
                dimension=self.config.dimension,
                n_clusters=100,
                metric=metric
            )
            index.train(self.vectors)
            index.add(self.vectors, ids=self.ids)
            index.nprobe = nprobe
            
            predicted = self._get_results_as_indices(index, k)
            recall = compute_recall(predicted, self.ground_truth[metric], k)
            
            search_times = [
                self._time_search(index, q, k) for q in self.queries[:20]
            ]
            
            tradeoff_data.append({
                "index": "ivf",
                "params": {"n_clusters": 100, "nprobe": nprobe},
                "recall": recall,
                "qps": 1.0 / np.mean(search_times),
                "build_time": 0
            })
        
        # Print trade-off table
        print("\n  Index          | Params                  | Recall | QPS")
        print("  " + "-" * 60)
        for entry in sorted(tradeoff_data, key=lambda x: x["recall"], reverse=True):
            params_str = str(entry["params"])[:20].ljust(20)
            print(f"  {entry['index']:14} | {params_str} | {entry['recall']:.4f} | {entry['qps']:.0f}")
        
        return {"k": k, "metric": metric, "data": tradeoff_data}
    
    def _time_search(self, index: Any, query: np.ndarray, k: int) -> float:
        """Time a single search operation."""
        start = time.perf_counter()
        index.search(query, k=k)
        return time.perf_counter() - start
    
    def run_all(self, metric: str = "euclidean") -> Dict[str, Any]:
        """Run all recall benchmarks."""
        all_results = {
            "baseline": self.benchmark_flat_baseline(metric),
            "hnsw": self.benchmark_hnsw_recall(metric),
            "ivf": self.benchmark_ivf_recall(metric),
            "pq": self.benchmark_pq_recall(metric),
            "tradeoff": self.benchmark_recall_vs_speed(metric),
        }
        return all_results
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "n_vectors": self.config.n_vectors,
                "n_queries": self.config.n_queries,
                "dimension": self.config.dimension,
                "k_values": self.config.k_values,
            },
            "results": [r.to_dict() for r in self.results]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self):
        """Print summary of recall results."""
        print("\n" + "=" * 70)
        print("RECALL BENCHMARK SUMMARY")
        print("=" * 70)
        
        # Group by index type
        by_index = {}
        for r in self.results:
            if r.index_type not in by_index:
                by_index[r.index_type] = []
            by_index[r.index_type].append(r)
        
        for index_type, results in by_index.items():
            print(f"\n{index_type.upper()}:")
            # Find best recall for each k
            for k in self.config.k_values:
                k_results = [r for r in results if r.k == k]
                if k_results:
                    best = max(k_results, key=lambda x: x.recall_at_k)
                    print(f"  k={k}: best recall={best.recall_at_k:.4f} "
                          f"(params: {best.index_params})")


def main():
    parser = argparse.ArgumentParser(description="VectorDB Recall Benchmarks")
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
        help="Number of vectors"
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Number of queries"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=128,
        help="Vector dimension"
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="k values to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/recall.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    if args.n_vectors:
        n_vectors = args.n_vectors
    else:
        preset = get_benchmark_config(args.size)
        n_vectors = preset["n_vectors"]
    
    config = RecallBenchmarkConfig(
        n_vectors=n_vectors,
        n_queries=args.n_queries,
        dimension=args.dimension,
        k_values=args.k,
        metrics=[args.metric]
    )
    
    print("=" * 60)
    print("VectorDB Recall Benchmark")
    print("=" * 60)
    print(f"Dataset: {config.n_vectors} vectors x {config.dimension} dims")
    print(f"Queries: {config.n_queries}")
    print(f"k values: {config.k_values}")
    print(f"Metric: {args.metric}")
    
    benchmark = RecallBenchmark(config)
    benchmark.run_all(args.metric)
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()