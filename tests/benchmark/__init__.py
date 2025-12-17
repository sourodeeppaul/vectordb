"""
Benchmark tests for VectorDB.

This module provides performance benchmarks for indexing,
search operations, and recall accuracy.

Usage:
    pytest tests/benchmark/ -v --benchmark
    python -m tests.benchmark.bench_search
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from contextlib import contextmanager
import json
import os
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    name: str
    operation: str
    dataset_size: int
    dimension: int
    
    # Timing metrics
    total_time: float
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    
    # Throughput metrics
    operations_per_second: float
    
    # Additional metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "operation": self.operation,
            "dataset_size": self.dataset_size,
            "dimension": self.dimension,
            "timing": {
                "total_time": self.total_time,
                "mean_time": self.mean_time,
                "std_time": self.std_time,
                "min_time": self.min_time,
                "max_time": self.max_time,
            },
            "throughput": {
                "ops_per_second": self.operations_per_second,
            },
            "extra_metrics": self.extra_metrics,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.name} ({self.operation}):\n"
            f"  Dataset: {self.dataset_size} vectors x {self.dimension} dims\n"
            f"  Mean time: {self.mean_time*1000:.3f} ms (±{self.std_time*1000:.3f})\n"
            f"  Throughput: {self.operations_per_second:.1f} ops/sec"
        )


@dataclass
class RecallResult:
    """Container for recall accuracy results."""
    
    name: str
    index_type: str
    dataset_size: int
    dimension: int
    k: int
    
    # Recall metrics
    recall_at_k: float
    recall_at_1: float
    
    # Timing
    build_time: float
    mean_search_time: float
    
    # Parameters
    index_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "index_type": self.index_type,
            "dataset_size": self.dataset_size,
            "dimension": self.dimension,
            "k": self.k,
            "recall": {
                "recall_at_k": self.recall_at_k,
                "recall_at_1": self.recall_at_1,
            },
            "timing": {
                "build_time": self.build_time,
                "mean_search_time": self.mean_search_time,
            },
            "index_params": self.index_params,
        }


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.elapsed = 0.0
        self._start = None
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


class BenchmarkRunner:
    """Utility class for running benchmarks."""
    
    def __init__(self, warmup_runs: int = 2, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        name: str,
        operation: str,
        func: Callable,
        dataset_size: int,
        dimension: int,
        **extra_metrics
    ) -> BenchmarkResult:
        """Run a benchmark and return results."""
        
        # Warmup
        for _ in range(self.warmup_runs):
            func()
        
        # Benchmark
        times = []
        for _ in range(self.benchmark_runs):
            with Timer() as t:
                func()
            times.append(t.elapsed)
        
        times = np.array(times)
        
        result = BenchmarkResult(
            name=name,
            operation=operation,
            dataset_size=dataset_size,
            dimension=dimension,
            total_time=np.sum(times),
            mean_time=np.mean(times),
            std_time=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times),
            operations_per_second=1.0 / np.mean(times),
            extra_metrics=extra_metrics,
        )
        
        self.results.append(result)
        return result
    
    def run_batch_benchmark(
        self,
        name: str,
        operation: str,
        func: Callable,
        batch_size: int,
        dataset_size: int,
        dimension: int,
        **extra_metrics
    ) -> BenchmarkResult:
        """Run a batch operation benchmark."""
        
        # Warmup
        for _ in range(self.warmup_runs):
            func()
        
        # Benchmark
        times = []
        for _ in range(self.benchmark_runs):
            with Timer() as t:
                func()
            times.append(t.elapsed)
        
        times = np.array(times)
        mean_per_item = np.mean(times) / batch_size
        
        result = BenchmarkResult(
            name=name,
            operation=operation,
            dataset_size=dataset_size,
            dimension=dimension,
            total_time=np.sum(times),
            mean_time=mean_per_item,
            std_time=np.std(times) / batch_size,
            min_time=np.min(times) / batch_size,
            max_time=np.max(times) / batch_size,
            operations_per_second=batch_size / np.mean(times),
            extra_metrics={"batch_size": batch_size, **extra_metrics},
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, filepath: str):
        """Save all results to JSON file."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in self.results],
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        for result in self.results:
            print(f"\n{result}")
        print("\n" + "=" * 60)


def generate_test_data(
    n_vectors: int,
    dimension: int,
    seed: int = 42
) -> np.ndarray:
    """Generate random test vectors."""
    np.random.seed(seed)
    return np.random.randn(n_vectors, dimension).astype(np.float32)


def generate_query_data(
    n_queries: int,
    dimension: int,
    seed: int = 123
) -> np.ndarray:
    """Generate random query vectors."""
    np.random.seed(seed)
    return np.random.randn(n_queries, dimension).astype(np.float32)


def compute_ground_truth(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str = "euclidean"
) -> np.ndarray:
    """Compute ground truth nearest neighbors using brute force."""
    from scipy.spatial.distance import cdist
    
    if metric == "cosine":
        # Normalize for cosine similarity
        data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)
        queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        distances = 1 - np.dot(queries_norm, data_norm.T)
    elif metric == "euclidean":
        distances = cdist(queries, data, metric="euclidean")
    elif metric == "dot":
        distances = -np.dot(queries, data.T)  # Negative for max similarity
    else:
        distances = cdist(queries, data, metric=metric)
    
    # Get indices of k smallest distances
    indices = np.argpartition(distances, k, axis=1)[:, :k]
    
    # Sort by actual distance
    for i in range(len(queries)):
        sorted_idx = np.argsort(distances[i, indices[i]])
        indices[i] = indices[i, sorted_idx]
    
    return indices


def compute_recall(
    predicted: List[List[int]],
    ground_truth: np.ndarray,
    k: int
) -> float:
    """Compute recall@k metric."""
    recalls = []
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        gt_set = set(gt[:k].tolist())
        if len(gt_set) > 0:
            recall = len(pred_set & gt_set) / len(gt_set)
            recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0


# Benchmark configuration presets
SMALL_DATASET = {"n_vectors": 1000, "dimension": 128, "n_queries": 100}
MEDIUM_DATASET = {"n_vectors": 10000, "dimension": 128, "n_queries": 100}
LARGE_DATASET = {"n_vectors": 100000, "dimension": 128, "n_queries": 100}
HIGH_DIM_DATASET = {"n_vectors": 10000, "dimension": 768, "n_queries": 100}


def get_benchmark_config(size: str = "medium") -> Dict[str, int]:
    """Get benchmark configuration by size name."""
    configs = {
        "small": SMALL_DATASET,
        "medium": MEDIUM_DATASET,
        "large": LARGE_DATASET,
        "high_dim": HIGH_DIM_DATASET,
    }
    return configs.get(size, MEDIUM_DATASET)