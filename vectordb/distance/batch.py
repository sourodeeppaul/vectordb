"""
Batch distance computation utilities.

Provides efficient methods for computing distances on large datasets,
including support for chunked processing to manage memory.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Callable
from numpy.typing import NDArray
import heapq

from .registry import get_metric_fn, get_metric
from .metrics import query_distances, normalize_vectors


class BatchDistanceCalculator:
    """
    Efficient batch distance calculator with memory management.
    
    Handles large-scale distance computations by processing in chunks
    to avoid memory overflow.
    
    Example:
        >>> calc = BatchDistanceCalculator(metric="euclidean", chunk_size=10000)
        >>> distances = calc.compute(query, database_vectors)
        >>> top_k = calc.top_k(query, database_vectors, k=10)
    """
    
    def __init__(
        self, 
        metric: str = "euclidean",
        chunk_size: int = 10000,
        normalize: bool = False,
    ):
        """
        Initialize batch calculator.
        
        Args:
            metric: Distance metric to use
            chunk_size: Number of vectors to process at once
            normalize: Whether to normalize vectors before computing
        """
        self.metric = metric
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.metric_info = get_metric(metric)
    
    def compute(
        self, 
        query: NDArray, 
        collection: NDArray,
        return_indices: bool = False,
    ) -> NDArray | Tuple[NDArray, NDArray]:
        """
        Compute distances from query to all vectors in collection.
        
        Args:
            query: Query vector of shape (d,) or (n_queries, d)
            collection: Collection of shape (n_vectors, d)
            return_indices: If True, also return sorted indices
            
        Returns:
            Distances array, optionally with sorted indices
        """
        # Normalize if required
        if self.normalize:
            query = self._normalize(query)
            collection = normalize_vectors(collection)
        
        # Handle single query vs batch queries
        if query.ndim == 1:
            return self._compute_single(query, collection, return_indices)
        else:
            return self._compute_batch(query, collection, return_indices)
    
    def _compute_single(
        self,
        query: NDArray,
        collection: NDArray,
        return_indices: bool,
    ) -> NDArray | Tuple[NDArray, NDArray]:
        """Compute distances for a single query."""
        n_vectors = len(collection)
        
        if n_vectors <= self.chunk_size:
            # Process all at once
            distances = query_distances(query, collection, self.metric)
        else:
            # Process in chunks
            distances = np.zeros(n_vectors, dtype=np.float32)
            
            for start in range(0, n_vectors, self.chunk_size):
                end = min(start + self.chunk_size, n_vectors)
                chunk = collection[start:end]
                distances[start:end] = query_distances(query, chunk, self.metric)
        
        if return_indices:
            indices = np.argsort(distances)
            return distances, indices
        
        return distances
    
    def _compute_batch(
        self,
        queries: NDArray,
        collection: NDArray,
        return_indices: bool,
    ) -> NDArray | Tuple[NDArray, NDArray]:
        """Compute distances for multiple queries."""
        n_queries = len(queries)
        n_vectors = len(collection)
        
        distances = np.zeros((n_queries, n_vectors), dtype=np.float32)
        
        for i, query in enumerate(queries):
            distances[i] = self._compute_single(query, collection, False)
        
        if return_indices:
            indices = np.argsort(distances, axis=1)
            return distances, indices
        
        return distances
    
    def _normalize(self, vectors: NDArray) -> NDArray:
        """Normalize vectors."""
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            if norm > 1e-8:
                return vectors / norm
            return vectors
        return normalize_vectors(vectors)
    
    def top_k(
        self,
        query: NDArray,
        collection: NDArray,
        k: int,
        return_distances: bool = True,
    ) -> Tuple[NDArray, Optional[NDArray]]:
        """
        Find top-k nearest neighbors.
        
        Uses efficient partial sorting instead of full sort.
        
        Args:
            query: Query vector
            collection: Collection of vectors
            k: Number of nearest neighbors
            return_distances: Whether to return distances
            
        Returns:
            Tuple of (indices, distances) or just indices
        """
        # Normalize if required
        if self.normalize:
            query = self._normalize(query)
            collection = normalize_vectors(collection)
        
        n_vectors = len(collection)
        k = min(k, n_vectors)
        
        if n_vectors <= self.chunk_size:
            # Compute all distances and use argpartition
            distances = query_distances(query, collection, self.metric)
            
            if k < n_vectors:
                # Partial sort for efficiency
                partition_indices = np.argpartition(distances, k)[:k]
                top_k_indices = partition_indices[np.argsort(distances[partition_indices])]
            else:
                top_k_indices = np.argsort(distances)
            
            if return_distances:
                return top_k_indices, distances[top_k_indices]
            return top_k_indices, None
        
        else:
            # Use heap for streaming top-k
            return self._top_k_chunked(query, collection, k, return_distances)
    
    def _top_k_chunked(
        self,
        query: NDArray,
        collection: NDArray,
        k: int,
        return_distances: bool,
    ) -> Tuple[NDArray, Optional[NDArray]]:
        """Find top-k using chunked processing with heap."""
        # Max heap to keep track of top-k (use negative for max-heap behavior)
        heap: List[Tuple[float, int]] = []
        
        n_vectors = len(collection)
        
        for start in range(0, n_vectors, self.chunk_size):
            end = min(start + self.chunk_size, n_vectors)
            chunk = collection[start:end]
            
            distances = query_distances(query, chunk, self.metric)
            
            for i, dist in enumerate(distances):
                global_idx = start + i
                
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, global_idx))
                elif dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, global_idx))
        
        # Extract results
        results = sorted([(-d, idx) for d, idx in heap])
        
        indices = np.array([idx for _, idx in results], dtype=np.int64)
        
        if return_distances:
            distances = np.array([d for d, _ in results], dtype=np.float32)
            return indices, distances
        
        return indices, None


def compute_all_distances(
    query: NDArray,
    collection: NDArray,
    metric: str = "euclidean",
) -> NDArray:
    """
    Convenience function to compute all distances.
    
    Args:
        query: Query vector
        collection: Collection of vectors
        metric: Distance metric
        
    Returns:
        Array of distances
    """
    return query_distances(query, collection, metric)


def compute_top_k(
    query: NDArray,
    collection: NDArray,
    k: int,
    metric: str = "euclidean",
) -> Tuple[NDArray, NDArray]:
    """
    Convenience function to find top-k nearest neighbors.
    
    Args:
        query: Query vector
        collection: Collection of vectors
        k: Number of neighbors
        metric: Distance metric
        
    Returns:
        Tuple of (indices, distances)
    """
    calc = BatchDistanceCalculator(metric=metric)
    return calc.top_k(query, collection, k)


def compute_pairwise_matrix(
    vectors: NDArray,
    metric: str = "euclidean",
    chunk_size: int = 1000,
) -> NDArray:
    """
    Compute full pairwise distance matrix.
    
    Args:
        vectors: Array of shape (n, d)
        metric: Distance metric
        chunk_size: Chunk size for processing
        
    Returns:
        Distance matrix of shape (n, n)
    """
    from .metrics import query_distances_matrix
    return query_distances_matrix(vectors, metric)


def find_nearest_neighbors(
    vectors: NDArray,
    k: int,
    metric: str = "euclidean",
    include_self: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Find k-nearest neighbors for each vector in the dataset.
    
    Args:
        vectors: Array of shape (n, d)
        k: Number of neighbors per vector
        metric: Distance metric
        include_self: Whether to include self as a neighbor
        
    Returns:
        Tuple of (indices, distances) each of shape (n, k)
    """
    n = len(vectors)
    actual_k = k + 1 if not include_self else k
    
    calc = BatchDistanceCalculator(metric=metric)
    
    all_indices = np.zeros((n, k), dtype=np.int64)
    all_distances = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        indices, distances = calc.top_k(vectors[i], vectors, actual_k)
        
        if not include_self:
            # Remove self (should be at index 0 with distance 0)
            mask = indices != i
            indices = indices[mask][:k]
            distances = distances[mask][:k]
        
        all_indices[i] = indices
        all_distances[i] = distances
    
    return all_indices, all_distances