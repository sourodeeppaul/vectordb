"""
SIMD-optimized distance calculations.

This module provides vectorized distance functions using NumPy
with optional Numba JIT compilation for enhanced performance.

The functions here are optimized versions of the base metrics
that leverage SIMD instructions through NumPy's vectorized operations.
"""

import numpy as np
from typing import Optional

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create no-op decorator if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, parallel=True, fastmath=True)
def simd_euclidean_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized batch Euclidean distance calculation.
    
    Computes Euclidean distance from query to all vectors using
    parallel loops when Numba is available.
    
    Args:
        query: Query vector of shape (d,)
        vectors: Matrix of vectors of shape (n, d)
        
    Returns:
        Array of distances of shape (n,)
    """
    n = vectors.shape[0]
    distances = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        diff = query - vectors[i]
        distances[i] = np.sqrt(np.sum(diff * diff))
    
    return distances


@jit(nopython=True, parallel=True, fastmath=True)
def simd_dot_product_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized batch dot product calculation.
    
    Args:
        query: Query vector of shape (d,)
        vectors: Matrix of vectors of shape (n, d)
        
    Returns:
        Array of dot products of shape (n,)
    """
    n = vectors.shape[0]
    results = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        results[i] = np.dot(query, vectors[i])
    
    return results


@jit(nopython=True, parallel=True, fastmath=True)
def simd_cosine_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized batch cosine distance calculation.
    
    Args:
        query: Query vector of shape (d,) (should be normalized)
        vectors: Matrix of vectors of shape (n, d) (should be normalized)
        
    Returns:
        Array of cosine distances of shape (n,)
    """
    n = vectors.shape[0]
    distances = np.empty(n, dtype=np.float32)
    
    query_norm = np.sqrt(np.sum(query * query))
    
    for i in prange(n):
        vec = vectors[i]
        vec_norm = np.sqrt(np.sum(vec * vec))
        dot = np.dot(query, vec)
        similarity = dot / (query_norm * vec_norm + 1e-10)
        distances[i] = 1.0 - similarity
    
    return distances


@jit(nopython=True, parallel=True, fastmath=True)
def simd_manhattan_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized batch Manhattan distance calculation.
    
    Args:
        query: Query vector of shape (d,)
        vectors: Matrix of vectors of shape (n, d)
        
    Returns:
        Array of distances of shape (n,)
    """
    n = vectors.shape[0]
    distances = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        distances[i] = np.sum(np.abs(query - vectors[i]))
    
    return distances


def numpy_euclidean_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    NumPy vectorized Euclidean distance (fallback when Numba unavailable).
    
    Uses broadcasting for efficient computation.
    """
    diff = vectors - query
    return np.sqrt(np.sum(diff * diff, axis=1))


def numpy_cosine_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    NumPy vectorized cosine distance (fallback when Numba unavailable).
    """
    query_norm = np.linalg.norm(query)
    vectors_norm = np.linalg.norm(vectors, axis=1)
    similarity = np.dot(vectors, query) / (query_norm * vectors_norm + 1e-10)
    return 1.0 - similarity


def numpy_dot_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    NumPy vectorized dot product (fallback when Numba unavailable).
    """
    return np.dot(vectors, query)


def get_optimized_distance_fn(metric: str, use_numba: Optional[bool] = None):
    """
    Get the most optimized distance function available.
    
    Args:
        metric: Distance metric name
        use_numba: Force Numba usage (None for auto-detect)
        
    Returns:
        Optimized distance function
    """
    if use_numba is None:
        use_numba = HAS_NUMBA
    
    if use_numba and HAS_NUMBA:
        functions = {
            "euclidean": simd_euclidean_batch,
            "cosine": simd_cosine_batch,
            "dot": simd_dot_product_batch,
            "manhattan": simd_manhattan_batch,
        }
    else:
        functions = {
            "euclidean": numpy_euclidean_batch,
            "cosine": numpy_cosine_batch,
            "dot": numpy_dot_batch,
            "manhattan": lambda q, v: np.sum(np.abs(v - q), axis=1),
        }
    
    if metric not in functions:
        raise ValueError(f"Unknown metric: {metric}")
    
    return functions[metric]


# Export availability flag
__all__ = [
    "HAS_NUMBA",
    "simd_euclidean_batch",
    "simd_dot_product_batch",
    "simd_cosine_batch",
    "simd_manhattan_batch",
    "numpy_euclidean_batch",
    "numpy_cosine_batch",
    "numpy_dot_batch",
    "get_optimized_distance_fn",
]
