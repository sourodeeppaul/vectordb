"""
Product Quantization (PQ) Index Implementation.

PQ compresses vectors by dividing them into subvectors and
quantizing each subvector independently. This enables:
- Significant memory reduction (up to 97%)
- Fast distance computation using lookup tables
- Approximate nearest neighbor search

Reference:
    Jegou, H., Douze, M., & Schmid, C. (2011).
    "Product quantization for nearest neighbor search."
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Iterator
import time

from .base import BaseIndex, IndexConfig, IndexStats, SearchResult, IndexType, FilterFunction


@dataclass
class PQConfig(IndexConfig):
    """Configuration for PQ index."""
    
    # Number of subvectors (must divide dimension evenly)
    n_subvectors: int = 8
    
    # Bits per subvector (8 = 256 centroids per subvector)
    n_bits: int = 8
    
    # Training iterations
    n_iter: int = 20
    
    # Training samples
    n_training_samples: int = 10000
    
    # Random seed
    seed: Optional[int] = None
    
    @property
    def n_centroids(self) -> int:
        """Number of centroids per subvector."""
        return 2 ** self.n_bits


class PQIndex(BaseIndex):
    """
    Product Quantization Index.
    
    Compresses vectors for memory-efficient approximate search.
    
    Example:
        >>> index = PQIndex(dimension=128, n_subvectors=8, n_bits=8)
        >>> index.train(training_vectors)
        >>> index.add_batch(ids, vectors)
        >>> results = index.search(query, k=10)
    
    Memory Usage:
        - Original: n * d * 4 bytes (float32)
        - Compressed: n * m bytes (uint8, m=n_subvectors)
        - Reduction: d * 4 / m (e.g., 128*4/8 = 64x)
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "euclidean",
        n_subvectors: int = 8,
        n_bits: int = 8,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(dimension, metric, **kwargs)
        
        if dimension % n_subvectors != 0:
            raise ValueError(
                f"Dimension {dimension} must be divisible by n_subvectors {n_subvectors}"
            )
        
        self.config = PQConfig(
            dimension=dimension,
            metric=metric,
            n_subvectors=n_subvectors,
            n_bits=n_bits,
            seed=seed,
        )
        
        self._subvector_dim = dimension // n_subvectors
        self._n_centroids = 2 ** n_bits
        
        # Codebook: (n_subvectors, n_centroids, subvector_dim)
        self._codebook: Optional[NDArray] = None
        
        # Compressed codes: Dict[id -> uint8 array of length n_subvectors]
        self._codes: Dict[str, NDArray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._ids: List[str] = []
        
        self._rng = np.random.RandomState(seed)
        self._build_start = time.time()
    
    @property
    def index_type(self) -> IndexType:
        return IndexType.PQ
    
    @property
    def size(self) -> int:
        return len(self._ids)
    
    def train(self, vectors: NDArray) -> None:
        """
        Train the PQ codebook using k-means.
        
        Args:
            vectors: Training vectors (n, dimension)
        """
        vectors = self._validate_vectors(vectors)
        n_samples = len(vectors)
        
        # Initialize codebook
        self._codebook = np.zeros(
            (self.config.n_subvectors, self._n_centroids, self._subvector_dim),
            dtype=np.float32,
        )
        
        # Train each subquantizer
        for m in range(self.config.n_subvectors):
            start = m * self._subvector_dim
            end = (m + 1) * self._subvector_dim
            
            subvectors = vectors[:, start:end]
            
            # K-means for this subvector
            centroids = self._train_subquantizer(subvectors)
            self._codebook[m] = centroids
        
        self._is_trained = True
    
    def _train_subquantizer(self, subvectors: NDArray) -> NDArray:
        """Train k-means for a single subquantizer."""
        n_samples = len(subvectors)
        k = min(self._n_centroids, n_samples)
        
        # Initialize with random samples
        indices = self._rng.choice(n_samples, k, replace=False)
        centroids = subvectors[indices].copy()
        
        # K-means iterations
        for _ in range(self.config.n_iter):
            # Assign to nearest centroid
            distances = self._pairwise_l2(subvectors, centroids)
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)
            
            for i, c in enumerate(assignments):
                new_centroids[c] += subvectors[i]
                counts[c] += 1
            
            # Avoid division by zero
            counts = np.maximum(counts, 1)
            centroids = new_centroids / counts[:, np.newaxis]
        
        # Pad if fewer centroids than expected
        if k < self._n_centroids:
            full_centroids = np.zeros(
                (self._n_centroids, self._subvector_dim),
                dtype=np.float32,
            )
            full_centroids[:k] = centroids
            centroids = full_centroids
        
        return centroids
    
    def _pairwise_l2(self, X: NDArray, Y: NDArray) -> NDArray:
        """Compute pairwise squared L2 distances."""
        X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)
        Y_sqnorm = np.sum(Y ** 2, axis=1, keepdims=True)
        return X_sqnorm + Y_sqnorm.T - 2 * X @ Y.T
    
    def _encode(self, vector: NDArray) -> NDArray:
        """Encode a vector into PQ codes."""
        codes = np.zeros(self.config.n_subvectors, dtype=np.uint8)
        
        for m in range(self.config.n_subvectors):
            start = m * self._subvector_dim
            end = (m + 1) * self._subvector_dim
            
            subvector = vector[start:end]
            distances = np.sum((self._codebook[m] - subvector) ** 2, axis=1)
            codes[m] = np.argmin(distances)
        
        return codes
    
    def _decode(self, codes: NDArray) -> NDArray:
        """Decode PQ codes back to approximate vector."""
        vector = np.zeros(self._dimension, dtype=np.float32)
        
        for m in range(self.config.n_subvectors):
            start = m * self._subvector_dim
            end = (m + 1) * self._subvector_dim
            vector[start:end] = self._codebook[m, codes[m]]
        
        return vector
    
    def add(
        self,
        id: str,
        vector: NDArray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector."""
        if not self._is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        
        vector = self._validate_vector(vector)
        
        if id in self._codes:
            raise ValueError(f"ID '{id}' already exists")
        
        codes = self._encode(vector)
        self._codes[id] = codes
        self._metadata[id] = metadata or {}
        self._ids.append(id)
    
    def add_batch(
        self,
        ids: List[str],
        vectors: NDArray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Add multiple vectors."""
        vectors = self._validate_vectors(vectors)
        
        for i, id in enumerate(ids):
            self.add(id, vectors[i], metadata[i] if metadata else None)
        
        return len(ids)
    
    def search(
        self,
        query: NDArray,
        k: int = 10,
        filter_fn: Optional[FilterFunction] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
    ) -> List[SearchResult]:
        """
        Search using asymmetric distance computation.
        
        Uses lookup tables for fast distance computation.
        """
        if not self._is_trained:
            raise RuntimeError("Index must be trained")
        
        if self.size == 0:
            return []
        
        query = self._validate_vector(query)
        
        # Compute distance lookup tables
        # distance_tables[m, c] = distance from query subvector m to centroid c
        distance_tables = np.zeros(
            (self.config.n_subvectors, self._n_centroids),
            dtype=np.float32,
        )
        
        for m in range(self.config.n_subvectors):
            start = m * self._subvector_dim
            end = (m + 1) * self._subvector_dim
            
            query_sub = query[start:end]
            distance_tables[m] = np.sum(
                (self._codebook[m] - query_sub) ** 2,
                axis=1,
            )
        
        # Compute distances using lookup tables
        distances = []
        for id in self._ids:
            codes = self._codes[id]
            metadata = self._metadata.get(id, {})
            
            # Apply filter
            if filter_fn and not filter_fn(id, metadata):
                continue
            
            # Sum distances from lookup tables
            dist = sum(distance_tables[m, codes[m]] for m in range(self.config.n_subvectors))
            distances.append((np.sqrt(dist), id, metadata))
        
        # Sort and take top k
        distances.sort(key=lambda x: x[0])
        top_k = distances[:k]
        
        # Build results
        results = []
        for dist, id, metadata in top_k:
            vector = None
            if include_vectors:
                vector = self._decode(self._codes[id])
            
            results.append(SearchResult(
                id=id,
                distance=dist,
                score=1.0 / (1.0 + dist),
                vector=vector,
                metadata=metadata if include_metadata else None,
            ))
        
        return results
    
    def search_batch(
        self,
        queries: NDArray,
        k: int = 10,
        filter_fn: Optional[FilterFunction] = None,
    ) -> List[List[SearchResult]]:
        """Batch search."""
        return [self.search(q, k=k, filter_fn=filter_fn) for q in queries]
    
    def remove(self, id: str) -> bool:
        """Remove a vector."""
        if id not in self._codes:
            return False
        
        del self._codes[id]
        del self._metadata[id]
        self._ids.remove(id)
        return True
    
    def remove_batch(self, ids: List[str]) -> int:
        """Remove multiple vectors."""
        return sum(1 for id in ids if self.remove(id))
    
    def get(self, id: str) -> Optional[Tuple[NDArray, Dict[str, Any]]]:
        """Get decoded vector by ID."""
        if id not in self._codes:
            return None
        
        vector = self._decode(self._codes[id])
        metadata = self._metadata.get(id, {})
        return vector, metadata
    
    def contains(self, id: str) -> bool:
        return id in self._codes
    
    def clear(self) -> int:
        count = len(self._ids)
        self._codes.clear()
        self._metadata.clear()
        self._ids.clear()
        return count
    
    def stats(self) -> IndexStats:
        # Memory for codes
        code_memory = self.size * self.config.n_subvectors  # uint8
        # Memory for codebook
        codebook_memory = (
            self.config.n_subvectors *
            self._n_centroids *
            self._subvector_dim * 4
        )
        
        original_memory = self.size * self._dimension * 4
        compression_ratio = original_memory / max(1, code_memory + codebook_memory)
        
        return IndexStats(
            index_type="pq",
            dimension=self._dimension,
            metric=self._metric,
            vector_count=self.size,
            memory_bytes=code_memory + codebook_memory,
            is_trained=self._is_trained,
            build_time_seconds=time.time() - self._build_start,
            extra={
                "n_subvectors": self.config.n_subvectors,
                "n_bits": self.config.n_bits,
                "n_centroids": self._n_centroids,
                "compression_ratio": compression_ratio,
                "original_memory_bytes": original_memory,
            },
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_type": "pq",
            "dimension": self._dimension,
            "metric": self._metric,
            "config": {
                "n_subvectors": self.config.n_subvectors,
                "n_bits": self.config.n_bits,
            },
            "codebook": self._codebook.tolist() if self._codebook is not None else None,
            "codes": {id: codes.tolist() for id, codes in self._codes.items()},
            "metadata": self._metadata,
            "ids": self._ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PQIndex":
        config = data.get("config", {})
        index = cls(
            dimension=data["dimension"],
            metric=data["metric"],
            n_subvectors=config.get("n_subvectors", 8),
            n_bits=config.get("n_bits", 8),
        )
        
        if data.get("codebook"):
            index._codebook = np.array(data["codebook"], dtype=np.float32)
            index._is_trained = True
        
        for id, codes in data.get("codes", {}).items():
            index._codes[id] = np.array(codes, dtype=np.uint8)
        
        index._metadata = data.get("metadata", {})
        index._ids = data.get("ids", [])
        
        return index
    
    def iter_ids(self) -> Iterator[str]:
        return iter(self._ids)
    
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        for id in self._ids:
            vector = self._decode(self._codes[id])
            metadata = self._metadata.get(id, {})
            yield id, vector, metadata