"""
Flat (Brute-Force) Index Implementation.

The Flat index performs exact nearest neighbor search by
computing distances to all vectors. It provides 100% recall
but has O(n) search complexity.

Best for:
    - Small to medium datasets (< 100k vectors)
    - When exact results are required
    - As a baseline for benchmarking other indices
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    Iterator,
    Set,
)
import heapq

from .base import (
    BaseIndex,
    IndexConfig,
    IndexStats,
    SearchResult,
    IndexType,
    FilterFunction,
)
from ..distance import (
    query_distances,
    get_metric,
    pairwise_euclidean,
    pairwise_cosine,
)


@dataclass
class FlatIndexConfig(IndexConfig):
    """Configuration for Flat index."""
    
    # Whether to maintain vectors in a contiguous array
    # (faster search but slower add/remove)
    use_matrix: bool = True
    
    # Initial capacity for pre-allocation
    initial_capacity: int = 1000
    
    # Growth factor when expanding capacity
    growth_factor: float = 2.0
    
    # Whether to normalize vectors on insert
    normalize: bool = False
    
    # Batch size for distance computations
    batch_size: int = 10000


class FlatIndex(BaseIndex):
    """
    Flat (Brute-Force) Index.
    
    Performs exact k-NN search by computing distances to all vectors.
    Provides 100% recall with O(n) search complexity.
    
    Example:
        >>> index = FlatIndex(dimension=128, metric="cosine")
        >>> 
        >>> # Add vectors
        >>> index.add("id1", vector1, {"type": "document"})
        >>> index.add_batch(ids, vectors, metadata_list)
        >>> 
        >>> # Search
        >>> results = index.search(query, k=10)
        >>> for r in results:
        ...     print(f"{r.id}: {r.distance:.4f}")
        >>> 
        >>> # Search with filter
        >>> results = index.search(
        ...     query, 
        ...     k=10,
        ...     filter_fn=lambda id, meta: meta.get("type") == "document"
        ... )
    
    Thread Safety:
        - Read operations (search, get) are thread-safe
        - Write operations (add, remove) use internal locking
        - For bulk writes, use add_batch for better performance
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "euclidean",
        normalize: bool = False,
        use_matrix: bool = True,
        initial_capacity: int = 1000,
        **kwargs,
    ):
        """
        Initialize Flat index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric ("euclidean", "cosine", "dot", etc.)
            normalize: Whether to L2-normalize vectors on insert
            use_matrix: Use contiguous matrix storage (faster search)
            initial_capacity: Initial capacity for pre-allocation
        """
        super().__init__(dimension, metric, **kwargs)
        
        self.config = FlatIndexConfig(
            dimension=dimension,
            metric=metric,
            normalize=normalize,
            use_matrix=use_matrix,
            initial_capacity=initial_capacity,
        )
        
        # Storage
        self._ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Matrix storage (optional but faster)
        if use_matrix:
            self._vectors = np.zeros(
                (initial_capacity, dimension), 
                dtype=np.float32
            )
            self._capacity = initial_capacity
        else:
            self._vectors_dict: Dict[str, NDArray] = {}
        
        self._size = 0
        self._is_trained = True  # Flat index doesn't need training
        
        # Track build time
        self._build_start = time.time()
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def index_type(self) -> IndexType:
        return IndexType.FLAT
    
    @property
    def size(self) -> int:
        return self._size
    
    # =========================================================================
    # ADD OPERATIONS
    # =========================================================================
    
    def add(
        self,
        id: str,
        vector: NDArray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a vector to the index.
        
        Args:
            id: Unique identifier
            vector: Vector to add
            metadata: Optional metadata dictionary
            
        Raises:
            ValueError: If ID already exists or vector dimension mismatch
        """
        vector = self._validate_vector(vector)
        
        if self.config.normalize:
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                vector = vector / norm
        
        with self._lock:
            if id in self._id_to_index:
                raise ValueError(f"ID '{id}' already exists in index")
            
            # Store vector
            if self.config.use_matrix:
                # Expand if needed
                if self._size >= self._capacity:
                    self._expand_capacity()
                
                self._vectors[self._size] = vector
            else:
                self._vectors_dict[id] = vector.copy()
            
            # Store mapping and metadata
            self._ids.append(id)
            self._id_to_index[id] = self._size
            self._metadata[id] = metadata or {}
            self._size += 1
    
    def add_batch(
        self,
        ids: List[str],
        vectors: NDArray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add multiple vectors to the index.
        
        More efficient than calling add() repeatedly.
        
        Args:
            ids: List of unique identifiers
            vectors: Array of vectors (n, dimension)
            metadata: Optional list of metadata dicts
            
        Returns:
            Number of vectors added
        """
        vectors = self._validate_vectors(vectors)
        n = len(ids)
        
        if len(vectors) != n:
            raise ValueError(
                f"Number of ids ({n}) != number of vectors ({len(vectors)})"
            )
        
        if metadata is not None and len(metadata) != n:
            raise ValueError(
                f"Number of ids ({n}) != number of metadata ({len(metadata)})"
            )
        
        if self.config.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            vectors = vectors / norms
        
        with self._lock:
            # Check for duplicates
            for id in ids:
                if id in self._id_to_index:
                    raise ValueError(f"ID '{id}' already exists in index")
            
            # Expand capacity if needed
            if self.config.use_matrix:
                required = self._size + n
                while required > self._capacity:
                    self._expand_capacity()
                
                # Copy vectors
                self._vectors[self._size:self._size + n] = vectors
            else:
                for i, id in enumerate(ids):
                    self._vectors_dict[id] = vectors[i].copy()
            
            # Update mappings
            for i, id in enumerate(ids):
                self._ids.append(id)
                self._id_to_index[id] = self._size + i
                self._metadata[id] = metadata[i] if metadata else {}
            
            self._size += n
        
        return n
    
    def _expand_capacity(self) -> None:
        """Expand matrix capacity."""
        new_capacity = int(self._capacity * self.config.growth_factor)
        new_vectors = np.zeros((new_capacity, self._dimension), dtype=np.float32)
        new_vectors[:self._size] = self._vectors[:self._size]
        self._vectors = new_vectors
        self._capacity = new_capacity
    
    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================
    
    def search(
        self,
        query: NDArray,
        k: int = 10,
        filter_fn: Optional[FilterFunction] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
    ) -> List[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results
            filter_fn: Optional filter function(id, metadata) -> bool
            include_vectors: Include vectors in results
            include_metadata: Include metadata in results
            
        Returns:
            List of SearchResult, sorted by distance (ascending)
        """
        if self._size == 0:
            return []
        
        query = self._validate_vector(query)
        
        if self.config.normalize:
            norm = np.linalg.norm(query)
            if norm > 1e-8:
                query = query / norm
        
        k = min(k, self._size)
        
        # Get vectors
        if self.config.use_matrix:
            vectors = self._vectors[:self._size]
        else:
            vectors = np.array(
                [self._vectors_dict[id] for id in self._ids],
                dtype=np.float32
            )
        
        # Compute distances
        distances = query_distances(query, vectors, self._metric)
        
        # Apply filter if provided
        if filter_fn is not None:
            results = self._search_with_filter(
                distances, k, filter_fn, include_vectors, include_metadata
            )
        else:
            results = self._search_no_filter(
                distances, k, include_vectors, include_metadata
            )
        
        return results
    
    def _search_no_filter(
        self,
        distances: NDArray,
        k: int,
        include_vectors: bool,
        include_metadata: bool,
    ) -> List[SearchResult]:
        """Search without filtering."""
        # Use argpartition for efficiency when k << n
        if k < len(distances) // 2:
            # Partial sort
            partition_indices = np.argpartition(distances, k)[:k]
            top_k_distances = distances[partition_indices]
            sorted_order = np.argsort(top_k_distances)
            top_k_indices = partition_indices[sorted_order]
        else:
            # Full sort
            top_k_indices = np.argsort(distances)[:k]
        
        # Build results
        results = []
        max_dist = distances[top_k_indices[-1]] if len(top_k_indices) > 0 else 1.0
        
        for idx in top_k_indices:
            id = self._ids[idx]
            dist = float(distances[idx])
            
            # Compute score (1 / (1 + distance))
            score = 1.0 / (1.0 + dist)
            
            result = SearchResult(
                id=id,
                distance=dist,
                score=score,
                vector=self._get_vector(id) if include_vectors else None,
                metadata=self._metadata.get(id, {}).copy() if include_metadata else None,
            )
            results.append(result)
        
        return results
    
    def _search_with_filter(
        self,
        distances: NDArray,
        k: int,
        filter_fn: FilterFunction,
        include_vectors: bool,
        include_metadata: bool,
    ) -> List[SearchResult]:
        """Search with filtering using heap."""
        # Use heap to maintain top-k while filtering
        heap: List[Tuple[float, int]] = []
        
        for idx in range(len(distances)):
            id = self._ids[idx]
            metadata = self._metadata.get(id, {})
            
            if not filter_fn(id, metadata):
                continue
            
            dist = distances[idx]
            
            if len(heap) < k:
                heapq.heappush(heap, (-dist, idx))  # Max heap
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, idx))
        
        # Sort by distance
        heap.sort(key=lambda x: -x[0])
        
        # Build results
        results = []
        for neg_dist, idx in heap:
            id = self._ids[idx]
            dist = -neg_dist
            score = 1.0 / (1.0 + dist)
            
            result = SearchResult(
                id=id,
                distance=dist,
                score=score,
                vector=self._get_vector(id) if include_vectors else None,
                metadata=self._metadata.get(id, {}).copy() if include_metadata else None,
            )
            results.append(result)
        
        return results
    
    def search_batch(
        self,
        queries: NDArray,
        k: int = 10,
        filter_fn: Optional[FilterFunction] = None,
    ) -> List[List[SearchResult]]:
        """
        Search with multiple queries.
        
        Args:
            queries: Array of query vectors (n, dimension)
            k: Number of results per query
            filter_fn: Optional filter function
            
        Returns:
            List of result lists, one per query
        """
        queries = self._validate_vectors(queries)
        
        if self.config.normalize:
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            queries = queries / norms
        
        return [
            self.search(q, k=k, filter_fn=filter_fn)
            for q in queries
        ]
    
    def search_by_id(
        self,
        id: str,
        k: int = 10,
        exclude_self: bool = True,
        filter_fn: Optional[FilterFunction] = None,
    ) -> List[SearchResult]:
        """
        Find vectors similar to an existing vector by ID.
        
        Args:
            id: ID of the query vector
            k: Number of results
            exclude_self: Exclude the query vector from results
            filter_fn: Optional filter function
            
        Returns:
            List of SearchResult
            
        Raises:
            KeyError: If ID not found
        """
        if id not in self._id_to_index:
            raise KeyError(f"ID '{id}' not found in index")
        
        query = self._get_vector(id)
        
        if exclude_self:
            original_filter = filter_fn
            filter_fn = lambda vid, meta: vid != id and (
                original_filter is None or original_filter(vid, meta)
            )
        
        return self.search(query, k=k, filter_fn=filter_fn)
    
    # =========================================================================
    # RANGE SEARCH
    # =========================================================================
    
    def range_search(
        self,
        query: NDArray,
        radius: float,
        filter_fn: Optional[FilterFunction] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Find all vectors within a distance radius.
        
        Args:
            query: Query vector
            radius: Maximum distance
            filter_fn: Optional filter function
            max_results: Maximum number of results
            
        Returns:
            List of SearchResult within radius
        """
        if self._size == 0:
            return []
        
        query = self._validate_vector(query)
        
        if self.config.normalize:
            norm = np.linalg.norm(query)
            if norm > 1e-8:
                query = query / norm
        
        # Get vectors
        if self.config.use_matrix:
            vectors = self._vectors[:self._size]
        else:
            vectors = np.array(
                [self._vectors_dict[id] for id in self._ids],
                dtype=np.float32
            )
        
        # Compute distances
        distances = query_distances(query, vectors, self._metric)
        
        # Find vectors within radius
        results = []
        
        for idx in range(len(distances)):
            if distances[idx] > radius:
                continue
            
            id = self._ids[idx]
            metadata = self._metadata.get(id, {})
            
            if filter_fn and not filter_fn(id, metadata):
                continue
            
            dist = float(distances[idx])
            score = 1.0 / (1.0 + dist)
            
            result = SearchResult(
                id=id,
                distance=dist,
                score=score,
                metadata=metadata.copy(),
            )
            results.append(result)
            
            if max_results and len(results) >= max_results:
                break
        
        # Sort by distance
        results.sort(key=lambda r: r.distance)
        
        return results
    
    # =========================================================================
    # REMOVE OPERATIONS
    # =========================================================================
    
    def remove(self, id: str) -> bool:
        """
        Remove a vector from the index.
        
        Args:
            id: Vector ID to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if id not in self._id_to_index:
                return False
            
            idx = self._id_to_index[id]
            
            # Remove from storage
            if self.config.use_matrix:
                # Swap with last element and shrink
                if idx < self._size - 1:
                    last_id = self._ids[-1]
                    self._vectors[idx] = self._vectors[self._size - 1]
                    self._id_to_index[last_id] = idx
                    self._ids[idx] = last_id
            else:
                del self._vectors_dict[id]
            
            # Update mappings
            del self._id_to_index[id]
            del self._metadata[id]
            self._ids.pop()
            self._size -= 1
            
            return True
    
    def remove_batch(self, ids: List[str]) -> int:
        """
        Remove multiple vectors.
        
        Args:
            ids: List of vector IDs to remove
            
        Returns:
            Number of vectors removed
        """
        count = 0
        for id in ids:
            if self.remove(id):
                count += 1
        return count
    
    # =========================================================================
    # GET OPERATIONS
    # =========================================================================
    
    def get(self, id: str) -> Optional[Tuple[NDArray, Dict[str, Any]]]:
        """
        Get a vector by ID.
        
        Args:
            id: Vector ID
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        if id not in self._id_to_index:
            return None
        
        vector = self._get_vector(id)
        metadata = self._metadata.get(id, {}).copy()
        
        return vector, metadata
    
    def _get_vector(self, id: str) -> NDArray:
        """Get vector by ID (internal)."""
        if self.config.use_matrix:
            idx = self._id_to_index[id]
            return self._vectors[idx].copy()
        else:
            return self._vectors_dict[id].copy()
    
    def get_vectors(self, ids: List[str]) -> Dict[str, Optional[NDArray]]:
        """
        Get multiple vectors by ID.
        
        Args:
            ids: List of vector IDs
            
        Returns:
            Dictionary mapping ID to vector (or None if not found)
        """
        return {
            id: self._get_vector(id) if id in self._id_to_index else None
            for id in ids
        }
    
    def contains(self, id: str) -> bool:
        """Check if vector ID exists."""
        return id in self._id_to_index
    
    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================
    
    def update(
        self,
        id: str,
        vector: Optional[NDArray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing vector.
        
        Args:
            id: Vector ID
            vector: New vector (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            if id not in self._id_to_index:
                return False
            
            idx = self._id_to_index[id]
            
            if vector is not None:
                vector = self._validate_vector(vector)
                
                if self.config.normalize:
                    norm = np.linalg.norm(vector)
                    if norm > 1e-8:
                        vector = vector / norm
                
                if self.config.use_matrix:
                    self._vectors[idx] = vector
                else:
                    self._vectors_dict[id] = vector.copy()
            
            if metadata is not None:
                self._metadata[id] = metadata
            
            return True
    
    # =========================================================================
    # UTILITY OPERATIONS
    # =========================================================================
    
    def clear(self) -> int:
        """
        Remove all vectors from the index.
        
        Returns:
            Number of vectors removed
        """
        with self._lock:
            count = self._size
            
            self._ids.clear()
            self._id_to_index.clear()
            self._metadata.clear()
            
            if self.config.use_matrix:
                self._vectors = np.zeros(
                    (self.config.initial_capacity, self._dimension),
                    dtype=np.float32
                )
                self._capacity = self.config.initial_capacity
            else:
                self._vectors_dict.clear()
            
            self._size = 0
            
            return count
    
    def stats(self) -> IndexStats:
        """Get index statistics."""
        # Calculate memory usage
        if self.config.use_matrix:
            vector_memory = self._vectors.nbytes
        else:
            vector_memory = sum(
                v.nbytes for v in self._vectors_dict.values()
            )
        
        # Estimate metadata memory
        metadata_memory = sum(
            len(str(m)) for m in self._metadata.values()
        )
        
        # Estimate ID memory
        id_memory = sum(len(id) for id in self._ids) * 2  # UTF-8 overhead
        
        return IndexStats(
            index_type=self.index_type.value,
            dimension=self._dimension,
            metric=self._metric,
            vector_count=self._size,
            memory_bytes=vector_memory + metadata_memory + id_memory,
            is_trained=True,
            build_time_seconds=time.time() - self._build_start,
            extra={
                "use_matrix": self.config.use_matrix,
                "normalize": self.config.normalize,
                "capacity": self._capacity if self.config.use_matrix else self._size,
            },
        )
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize index to dictionary."""
        if self.config.use_matrix:
            vectors = self._vectors[:self._size].tolist()
        else:
            vectors = [
                self._vectors_dict[id].tolist() 
                for id in self._ids
            ]
        
        return {
            "index_type": self.index_type.value,
            "dimension": self._dimension,
            "metric": self._metric,
            "config": {
                "normalize": self.config.normalize,
                "use_matrix": self.config.use_matrix,
                "initial_capacity": self.config.initial_capacity,
            },
            "ids": self._ids.copy(),
            "vectors": vectors,
            "metadata": {id: meta.copy() for id, meta in self._metadata.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlatIndex":
        """Deserialize index from dictionary."""
        config = data.get("config", {})
        
        index = cls(
            dimension=data["dimension"],
            metric=data["metric"],
            normalize=config.get("normalize", False),
            use_matrix=config.get("use_matrix", True),
            initial_capacity=max(
                config.get("initial_capacity", 1000),
                len(data.get("ids", []))
            ),
        )
        
        # Add vectors
        ids = data.get("ids", [])
        vectors = np.array(data.get("vectors", []), dtype=np.float32)
        metadata_dict = data.get("metadata", {})
        
        if len(ids) > 0:
            metadata = [metadata_dict.get(id, {}) for id in ids]
            index.add_batch(ids, vectors, metadata)
        
        return index
    
    # =========================================================================
    # ITERATION
    # =========================================================================
    
    def iter_ids(self) -> Iterator[str]:
        """Iterate over all vector IDs."""
        return iter(self._ids)
    
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        """Iterate over all vectors."""
        for id in self._ids:
            vector = self._get_vector(id)
            metadata = self._metadata.get(id, {})
            yield id, vector, metadata
    
    def get_all_vectors(self) -> NDArray:
        """
        Get all vectors as a matrix.
        
        Returns:
            Array of shape (n, dimension)
        """
        if self.config.use_matrix:
            return self._vectors[:self._size].copy()
        else:
            return np.array(
                [self._vectors_dict[id] for id in self._ids],
                dtype=np.float32
            )
    
    def get_all_ids(self) -> List[str]:
        """Get all vector IDs."""
        return self._ids.copy()
    
    # =========================================================================
    # ANALYSIS & DIAGNOSTICS
    # =========================================================================
    
    def compute_distance_matrix(self) -> NDArray:
        """
        Compute pairwise distance matrix.
        
        Warning: O(n²) memory and computation.
        
        Returns:
            Distance matrix of shape (n, n)
        """
        vectors = self.get_all_vectors()
        
        if self._metric == "euclidean":
            return pairwise_euclidean(vectors)
        elif self._metric == "cosine":
            return pairwise_cosine(vectors)
        else:
            # Fallback
            n = len(vectors)
            result = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                result[i] = query_distances(vectors[i], vectors, self._metric)
            return result
    
    def find_duplicates(
        self,
        threshold: float = 1e-6,
    ) -> List[Tuple[str, str, float]]:
        """
        Find duplicate or near-duplicate vectors.
        
        Args:
            threshold: Maximum distance to consider as duplicate
            
        Returns:
            List of (id1, id2, distance) tuples
        """
        duplicates = []
        vectors = self.get_all_vectors()
        
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dist = float(query_distances(
                    vectors[i], 
                    vectors[j:j+1], 
                    self._metric
                )[0])
                
                if dist <= threshold:
                    duplicates.append((self._ids[i], self._ids[j], dist))
        
        return duplicates
    
    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> List[Tuple[str, NDArray, Dict[str, Any]]]:
        """
        Random sample of vectors.
        
        Args:
            n: Number of vectors to sample
            seed: Random seed
            
        Returns:
            List of (id, vector, metadata) tuples
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = min(n, self._size)
        indices = np.random.choice(self._size, n, replace=False)
        
        return [
            (self._ids[i], self._get_vector(self._ids[i]), self._metadata.get(self._ids[i], {}))
            for i in indices
        ]