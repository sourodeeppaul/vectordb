"""
Distance metrics for vector similarity search.

This module provides various distance and similarity functions
optimized for high-dimensional vector operations.

Supported Metrics:
    - euclidean: L2 distance (smaller = more similar)
    - cosine: Cosine distance (smaller = more similar)
    - dot: Negative dot product (smaller = more similar)
    - manhattan: L1 distance (smaller = more similar)
    - chebyshev: L∞ distance (smaller = more similar)
    - hamming: Hamming distance for binary vectors

Example:
    >>> from vectordb.distance import euclidean, cosine, get_metric
    >>> import numpy as np
    >>> 
    >>> a = np.array([1.0, 2.0, 3.0])
    >>> b = np.array([4.0, 5.0, 6.0])
    >>> 
    >>> # Direct function call
    >>> dist = euclidean(a, b)
    >>> 
    >>> # Using registry
    >>> metric_fn = get_metric("cosine")
    >>> dist = metric_fn(a, b)
"""

from .metrics import (
    # Single vector distances
    euclidean,
    euclidean_squared,
    cosine_distance,
    cosine_similarity,
    dot_product,
    negative_dot_product,
    manhattan,
    chebyshev,
    hamming,
    # Batch operations
    pairwise_euclidean,
    pairwise_cosine,
    pairwise_dot,
    pairwise_manhattan,
    # Query to collection
    query_distances,
)

from .registry import (
    DistanceMetric,
    get_metric,
    get_metric_fn,
    register_metric,
    list_metrics,
    is_similarity,
)

from .batch import (
    BatchDistanceCalculator,
    compute_all_distances,
    compute_top_k,
)

__all__ = [
    # Single vector functions
    "euclidean",
    "euclidean_squared",
    "cosine_distance",
    "cosine_similarity",
    "dot_product",
    "negative_dot_product",
    "manhattan",
    "chebyshev",
    "hamming",
    # Batch functions
    "pairwise_euclidean",
    "pairwise_cosine",
    "pairwise_dot",
    "pairwise_manhattan",
    "query_distances",
    # Registry
    "DistanceMetric",
    "get_metric",
    "get_metric_fn",
    "register_metric",
    "list_metrics",
    "is_similarity",
    # Batch calculator
    "BatchDistanceCalculator",
    "compute_all_distances",
    "compute_top_k",
]