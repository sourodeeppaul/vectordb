"""
Hybrid Index (IVF + PQ) Implementation.

Combines IVF clustering with PQ compression for
scalable approximate nearest neighbor search.

This is the most scalable index type, suitable for
billion-scale datasets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Iterator
from collections import defaultdict
import time

from .base import BaseIndex, IndexConfig, IndexStats, SearchResult, IndexType, FilterFunction
from .ivf import KMeans


@dataclass
class HybridConfig(IndexConfig):
    """Configuration for IVF-PQ hybrid index."""
    
    # IVF settings
    n_clusters: int = 100
    n_probe: int = 10
    
    # PQ settings
    n_subvectors: int = 8
    n_bits: int = 8
    
    # Training
    n_iter: int = 20
    seed: Optional[int] = None


class HybridIndex(BaseIndex):
    """
    IVF-PQ Hybrid Index.
    
    Uses IVF for coarse quantization (partitioning) and
    PQ for fine quantization (compression) within each cluster.
    
    Example:
        >>> index = HybridIndex(
        ...     dimension=128,
        ...     n_clusters=1000,
        ...     n_subvectors=8,
        ...     n_probe=10,
        ... )
        >>> index.train(training_vectors)
        >>> index.add_batch(ids, vectors)
        >>> results = index.search(query, k=10)
    
    Complexity:
        - Memory: O(n * m) where m = n_subvectors (vs O(n * d) for flat)
        - Search: O(n_probe * n/n_clusters)
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "euclidean",
        n_clusters: int = 100,
        n_probe: int = 10,
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
        
        self.config = HybridConfig(
            dimension=dimension,
            metric=metric,
            n_clusters=n_clusters,
            n_probe=n_probe,
            n_subvectors=n_subvectors,
            n_bits=n_bits,
            seed=seed,
        )
        
        self._subvector_dim = dimension // n_subvectors
        self._n_centroids_pq = 2 ** n_bits
        
        # IVF centroids
        self._coarse_centroids: Optional[NDArray] = None
        
        # PQ codebooks (one per cluster for residuals)
        # Using shared codebook for simplicity
        self._pq_codebook: Optional[NDArray] = None
        
        # Inverted lists: cluster_id -> list of (id, codes, metadata)
        self._inverted_lists: Dict[int, List[Tuple[str, NDArray, Dict]]] = defaultdict(list)
        
        # ID to cluster mapping
        self._id_to_cluster: Dict[str, int] = {}
        
        self._ids: List[str] = []
        self._size = 0
        
        self._rng = np.random.RandomState(seed)
        self._build_start = time.time()
    
    @property
    def index_type(self) -> IndexType:
        return IndexType.IVF_PQ
    
    @property
    def size(self) -> int:
        return self._size
    
    def set_n_probe(self, n_probe: int) -> None:
        """Set number of clusters to search."""
        self.config.n_probe = min(n_probe, self.config.n_clusters)
    
    def train(self, vectors: NDArray) -> None:
        """
        Train the hybrid index.
        
        1. Train IVF (coarse quantizer) with k-means
        2. Compute residuals
        3. Train PQ on residuals
        """
        vectors = self._validate_vectors(vectors)
        n_samples = len(vectors)
        
        # 1. Train coarse quantizer
        actual_clusters = min(self.config.n_clusters, n_samples)
        
        kmeans = KMeans(
            n_clusters=actual_clusters,
            n_iter=self.config.n_iter,
            seed=self.config.seed,
        )
        self._coarse_centroids = kmeans.fit(vectors)
        self.config.n_clusters = len(self._coarse_centroids)
        
        # 2. Compute residuals
        assignments = self._assign_coarse(vectors)
        residuals = vectors - self._coarse_centroids[assignments]
        
        # 3. Train PQ on residuals
        self._train_pq(residuals)
        
        # Initialize inverted lists
        self._inverted_lists = {i: [] for i in range(self.config.n_clusters)}
        
        self._is_trained = True
    
    def _train_pq(self, residuals: NDArray) -> None:
        """Train PQ codebook on residuals."""
        self._pq_codebook = np.zeros(
            (self.config.n_subvectors, self._n_centroids_pq, self._subvector_dim),
            dtype=np.float32,
        )
        
        for m in range(self.config.n_subvectors):
            start = m * self._subvector_dim
            end = (m + 1) * self._subvector_dim
            
            subvectors = residuals[:, start:end]
            self._pq_codebook[m] = self._train_subquantizer(subvectors)
    
    def _train_subquantizer(self, subvectors: NDArray) -> NDArray:
        """Train k-means for a single subquantizer."""
        n_samples = len(subvectors)
        k = min(self._n_centroids_pq, n_samples)
        
        indices = self._rng.choice(n_samples, k, replace=False)
        centroids = subvectors[indices].copy()
        
        for _ in range(self.config.n_iter):
            X_sqnorm = np.sum(subvectors ** 2, axis=1, keepdims=True)
            C_sqnorm = np.sum(centroids ** 2, axis=1, keepdims=True)
            distances = X_sqnorm + C_sqnorm.T - 2 * subvectors @ centroids.T
            assignments = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)
            
            for i, c in enumerate(assignments):
                new_centroids[c] += subvectors[i]
                counts[c] += 1
            
            counts = np.maximum(counts, 1)
            centroids = new_centroids / counts[:, np.newaxis]
        
        if k < self._n_centroids_pq:
            full = np.zeros((self._n_centroids_pq, self._subvector_dim), dtype=np.float32)
            full[:k] = centroids
            centroids = full
        
        return centroids
    
    def _assign_coarse(self, vectors: NDArray) -> NDArray:
        """Assign vectors to coarse clusters."""
        X_sqnorm = np.sum(vectors ** 2, axis=1, keepdims=True)
        C_sqnorm = np.sum(self._coarse_centroids ** 2, axis=1, keepdims=True)
        distances = X_sqnorm + C_sqnorm.T - 2 * vectors @ self._coarse_centroids.T
        return np.argmin(distances, axis=1)
    
    def _encode_pq(self, residual: NDArray) -> NDArray:
        """Encode residual into PQ codes."""
        codes = np.zeros(self.config.n_subvectors, dtype=np.uint8)
        
        for m in range(self.config.n_subvectors):
            start = m * self._subvector_dim
            end = (m + 1) * self._subvector_dim
            
            sub = residual[start:end]
            distances = np.sum((self._pq_codebook[m] - sub) ** 2, axis=1)
            codes[m] = np.argmin(distances)
        
        return codes
    
    def add(
        self,
        id: str,
        vector: NDArray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector."""
        if not self._is_trained:
            raise RuntimeError("Index must be trained")
        
        vector = self._validate_vector(vector)
        
        if id in self._id_to_cluster:
            raise ValueError(f"ID '{id}' already exists")
        
        # Assign to cluster
        cluster_id = int(self._assign_coarse(vector.reshape(1, -1))[0])
        
        # Compute residual and encode
        residual = vector - self._coarse_centroids[cluster_id]
        codes = self._encode_pq(residual)
        
        # Store
        self._inverted_lists[cluster_id].append((id, codes, metadata or {}))
        self._id_to_cluster[id] = cluster_id
        self._ids.append(id)
        self._size += 1
    
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
        n_probe: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search using IVF-PQ.
        
        1. Find nearest coarse centroids
        2. For each cluster, compute distances using PQ
        3. Return top-k overall
        """
        if not self._is_trained:
            raise RuntimeError("Index must be trained")
        
        if self._size == 0:
            return []
        
        query = self._validate_vector(query)
        n_probe = n_probe or self.config.n_probe
        
        # Find nearest coarse centroids
        query_reshaped = query.reshape(1, -1)
        C_sqnorm = np.sum(self._coarse_centroids ** 2, axis=1)
        Q_sqnorm = np.sum(query ** 2)
        coarse_distances = Q_sqnorm + C_sqnorm - 2 * (query @ self._coarse_centroids.T)
        nearest_clusters = np.argsort(coarse_distances)[:n_probe]
        
        # Search in each cluster
        candidates = []
        
        for cluster_id in nearest_clusters:
            centroid = self._coarse_centroids[cluster_id]
            residual_query = query - centroid
            
            # Build distance lookup tables
            distance_tables = np.zeros(
                (self.config.n_subvectors, self._n_centroids_pq),
                dtype=np.float32,
            )
            
            for m in range(self.config.n_subvectors):
                start = m * self._subvector_dim
                end = (m + 1) * self._subvector_dim
                
                q_sub = residual_query[start:end]
                distance_tables[m] = np.sum(
                    (self._pq_codebook[m] - q_sub) ** 2,
                    axis=1,
                )
            
            # Compute distances for all vectors in cluster
            for id, codes, metadata in self._inverted_lists[cluster_id]:
                if filter_fn and not filter_fn(id, metadata):
                    continue
                
                dist = sum(distance_tables[m, codes[m]] for m in range(self.config.n_subvectors))
                candidates.append((np.sqrt(dist), id, metadata))
        
        # Sort and take top k
        candidates.sort(key=lambda x: x[0])
        top_k = candidates[:k]
        
        # Build results
        results = []
        for dist, id, metadata in top_k:
            vector = None
            if include_vectors:
                vector = self._reconstruct(id)
            
            results.append(SearchResult(
                id=id,
                distance=dist,
                score=1.0 / (1.0 + dist),
                vector=vector,
                metadata=metadata if include_metadata else None,
            ))
        
        return results
    
    def _reconstruct(self, id: str) -> NDArray:
        """Reconstruct approximate vector from codes."""
        cluster_id = self._id_to_cluster[id]
        
        # Find codes
        codes = None
        for stored_id, stored_codes, _ in self._inverted_lists[cluster_id]:
            if stored_id == id:
                codes = stored_codes
                break
        
        if codes is None:
            raise ValueError(f"ID '{id}' not found")
        
        # Reconstruct residual
        residual = np.zeros(self._dimension, dtype=np.float32)
        for m in range(self.config.n_subvectors):
            start = m * self._subvector_dim
            end = (m + 1) * self._subvector_dim
            residual[start:end] = self._pq_codebook[m, codes[m]]
        
        # Add centroid
        return self._coarse_centroids[cluster_id] + residual
    
    def search_batch(
        self,
        queries: NDArray,
        k: int = 10,
        filter_fn: Optional[FilterFunction] = None,
        n_probe: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """Batch search."""
        return [
            self.search(q, k=k, filter_fn=filter_fn, n_probe=n_probe)
            for q in queries
        ]
    
    def remove(self, id: str) -> bool:
        """Remove a vector."""
        if id not in self._id_to_cluster:
            return False
        
        cluster_id = self._id_to_cluster[id]
        
        # Remove from inverted list
        self._inverted_lists[cluster_id] = [
            (i, c, m) for i, c, m in self._inverted_lists[cluster_id]
            if i != id
        ]
        
        del self._id_to_cluster[id]
        self._ids.remove(id)
        self._size -= 1
        
        return True
    
    def remove_batch(self, ids: List[str]) -> int:
        return sum(1 for id in ids if self.remove(id))
    
    def get(self, id: str) -> Optional[Tuple[NDArray, Dict[str, Any]]]:
        """Get reconstructed vector by ID."""
        if id not in self._id_to_cluster:
            return None
        
        cluster_id = self._id_to_cluster[id]
        
        for stored_id, codes, metadata in self._inverted_lists[cluster_id]:
            if stored_id == id:
                vector = self._reconstruct(id)
                return vector, metadata
        
        return None
    
    def contains(self, id: str) -> bool:
        return id in self._id_to_cluster
    
    def clear(self) -> int:
        count = self._size
        
        for cluster_id in self._inverted_lists:
            self._inverted_lists[cluster_id] = []
        
        self._id_to_cluster.clear()
        self._ids.clear()
        self._size = 0
        
        return count
    
    def stats(self) -> IndexStats:
        # Memory calculation
        code_memory = self._size * self.config.n_subvectors
        codebook_memory = (
            self.config.n_subvectors *
            self._n_centroids_pq *
            self._subvector_dim * 4
        )
        centroid_memory = self.config.n_clusters * self._dimension * 4
        
        total_memory = code_memory + codebook_memory + centroid_memory
        original_memory = self._size * self._dimension * 4
        
        return IndexStats(
            index_type="ivf_pq",
            dimension=self._dimension,
            metric=self._metric,
            vector_count=self._size,
            memory_bytes=total_memory,
            is_trained=self._is_trained,
            build_time_seconds=time.time() - self._build_start,
            extra={
                "n_clusters": self.config.n_clusters,
                "n_probe": self.config.n_probe,
                "n_subvectors": self.config.n_subvectors,
                "n_bits": self.config.n_bits,
                "compression_ratio": original_memory / max(1, total_memory),
            },
        )
    
    def to_dict(self) -> Dict[str, Any]:
        inverted_lists_data = {}
        for cluster_id, entries in self._inverted_lists.items():
            inverted_lists_data[str(cluster_id)] = [
                {"id": id, "codes": codes.tolist(), "metadata": meta}
                for id, codes, meta in entries
            ]
        
        return {
            "index_type": "ivf_pq",
            "dimension": self._dimension,
            "metric": self._metric,
            "config": {
                "n_clusters": self.config.n_clusters,
                "n_probe": self.config.n_probe,
                "n_subvectors": self.config.n_subvectors,
                "n_bits": self.config.n_bits,
            },
            "coarse_centroids": self._coarse_centroids.tolist() if self._coarse_centroids is not None else None,
            "pq_codebook": self._pq_codebook.tolist() if self._pq_codebook is not None else None,
            "inverted_lists": inverted_lists_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridIndex":
        config = data.get("config", {})
        
        index = cls(
            dimension=data["dimension"],
            metric=data["metric"],
            n_clusters=config.get("n_clusters", 100),
            n_probe=config.get("n_probe", 10),
            n_subvectors=config.get("n_subvectors", 8),
            n_bits=config.get("n_bits", 8),
        )
        
        if data.get("coarse_centroids"):
            index._coarse_centroids = np.array(data["coarse_centroids"], dtype=np.float32)
        
        if data.get("pq_codebook"):
            index._pq_codebook = np.array(data["pq_codebook"], dtype=np.float32)
        
        index._is_trained = index._coarse_centroids is not None
        
        # Restore inverted lists
        for cluster_id_str, entries in data.get("inverted_lists", {}).items():
            cluster_id = int(cluster_id_str)
            for entry in entries:
                codes = np.array(entry["codes"], dtype=np.uint8)
                index._inverted_lists[cluster_id].append(
                    (entry["id"], codes, entry.get("metadata", {}))
                )
                index._id_to_cluster[entry["id"]] = cluster_id
                index._ids.append(entry["id"])
                index._size += 1
        
        return index
    
    def iter_ids(self) -> Iterator[str]:
        return iter(self._ids)
    
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        for id in self._ids:
            result = self.get(id)
            if result:
                vector, metadata = result
                yield id, vector, metadata