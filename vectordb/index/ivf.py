"""
IVF (Inverted File) Index Implementation.

IVF is a clustering-based approximate nearest neighbor algorithm that:
1. Partitions the vector space into clusters using k-means
2. Assigns each vector to its nearest cluster (inverted list)
3. At search time, only searches the most relevant clusters

Key Features:
    - Sub-linear search complexity O(n/k * n_probe)
    - Tunable accuracy vs speed tradeoff (n_probe)
    - Memory efficient
    - Supports incremental additions after training

Reference:
    Jegou, H., Douze, M., & Schmid, C. (2011).
    "Product quantization for nearest neighbor search."
    IEEE transactions on pattern analysis and machine intelligence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import threading
import time
import heapq
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    Iterator,
    Set,
    NamedTuple,
)
from collections import defaultdict

from .base import (
    BaseIndex,
    IndexConfig,
    IndexStats,
    SearchResult,
    IndexType,
    FilterFunction,
)
from ..distance import (
    get_metric_fn,
    query_distances,
    pairwise_euclidean,
    pairwise_cosine,
)


class VectorEntry(NamedTuple):
    """Entry in an inverted list."""
    id: str
    vector: NDArray
    metadata: Dict[str, Any]


@dataclass
class IVFConfig(IndexConfig):
    """Configuration for IVF index."""
    
    # Number of clusters (centroids)
    n_clusters: int = 100
    
    # Number of clusters to search (higher = better recall, slower)
    n_probe: int = 10
    
    # K-means parameters
    n_iter: int = 20              # K-means iterations
    n_redo: int = 1               # Number of k-means runs (best is kept)
    max_points_per_centroid: int = 256  # For sampling during training
    min_points_per_centroid: int = 1
    
    # Whether to normalize vectors
    normalize: bool = False
    
    # Random seed
    seed: Optional[int] = None
    
    # Verbose training
    verbose: bool = False
    
    def __post_init__(self):
        super().validate()
        
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}")
        
        if self.n_probe < 1:
            raise ValueError(f"n_probe must be >= 1, got {self.n_probe}")
        
        if self.n_probe > self.n_clusters:
            self.n_probe = self.n_clusters


class KMeans:
    """
    K-Means clustering implementation.
    
    Used to train IVF centroids.
    """
    
    def __init__(
        self,
        n_clusters: int,
        n_iter: int = 20,
        n_redo: int = 1,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_redo = n_redo
        self.seed = seed
        self.verbose = verbose
        
        self.centroids: Optional[NDArray] = None
        self._rng = np.random.RandomState(seed)
    
    def fit(self, vectors: NDArray) -> NDArray:
        """
        Fit k-means to vectors.
        
        Args:
            vectors: Training vectors (n, d)
            
        Returns:
            Centroids (n_clusters, d)
        """
        n_samples, dimension = vectors.shape
        
        if n_samples < self.n_clusters:
            # Not enough samples, use all as centroids
            self.centroids = vectors.copy()
            return self.centroids
        
        best_centroids = None
        best_inertia = float('inf')
        
        for redo in range(self.n_redo):
            centroids = self._fit_once(vectors)
            
            # Compute inertia (sum of squared distances to nearest centroid)
            assignments = self._assign(vectors, centroids)
            inertia = 0.0
            for i, c in enumerate(assignments):
                inertia += np.sum((vectors[i] - centroids[c]) ** 2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
            
            if self.verbose:
                print(f"  K-Means redo {redo + 1}/{self.n_redo}: inertia={inertia:.2f}")
        
        self.centroids = best_centroids
        return self.centroids
    
    def _fit_once(self, vectors: NDArray) -> NDArray:
        """Single k-means run."""
        n_samples, dimension = vectors.shape
        
        # Initialize centroids using k-means++
        centroids = self._init_centroids_pp(vectors)
        
        for iteration in range(self.n_iter):
            # Assign points to clusters
            assignments = self._assign(vectors, centroids)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(self.n_clusters)
            
            for i, c in enumerate(assignments):
                new_centroids[c] += vectors[i]
                counts[c] += 1
            
            # Handle empty clusters
            for c in range(self.n_clusters):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                else:
                    # Reinitialize empty cluster with random point
                    new_centroids[c] = vectors[self._rng.randint(n_samples)]
            
            # Check convergence
            diff = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            
            if self.verbose and (iteration + 1) % 5 == 0:
                print(f"    Iteration {iteration + 1}/{self.n_iter}: diff={diff:.6f}")
            
            if diff < 1e-6:
                break
        
        return centroids
    
    def _init_centroids_pp(self, vectors: NDArray) -> NDArray:
        """Initialize centroids using k-means++."""
        n_samples, dimension = vectors.shape
        centroids = np.zeros((self.n_clusters, dimension), dtype=vectors.dtype)
        
        # First centroid: random point
        idx = self._rng.randint(n_samples)
        centroids[0] = vectors[idx]
        
        # Remaining centroids
        for k in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = np.min(
                pairwise_euclidean(vectors, centroids[:k]) ** 2,
                axis=1
            )
            
            # Sample proportional to distance squared
            probs = distances / distances.sum()
            idx = self._rng.choice(n_samples, p=probs)
            centroids[k] = vectors[idx]
        
        return centroids
    
    def _assign(self, vectors: NDArray, centroids: NDArray) -> NDArray:
        """Assign vectors to nearest centroids."""
        distances = pairwise_euclidean(vectors, centroids)
        return np.argmin(distances, axis=1)
    
    def predict(self, vectors: NDArray) -> NDArray:
        """Predict cluster assignments."""
        if self.centroids is None:
            raise RuntimeError("K-Means not fitted")
        return self._assign(vectors, self.centroids)


class IVFIndex(BaseIndex):
    """
    IVF (Inverted File) Index.
    
    A clustering-based index that partitions vectors into clusters
    and searches only the most relevant clusters at query time.
    
    Example:
        >>> # Create and train index
        >>> index = IVFIndex(dimension=128, n_clusters=100, n_probe=10)
        >>> index.train(training_vectors)
        >>> 
        >>> # Add vectors
        >>> for i in range(100000):
        ...     index.add(f"vec{i}", vectors[i], {"idx": i})
        >>> 
        >>> # Search
        >>> results = index.search(query, k=10)
        >>> 
        >>> # Tune recall vs speed
        >>> index.set_n_probe(20)  # Higher = better recall, slower
        >>> results = index.search(query, k=10)
    
    Parameters:
        n_clusters: Number of clusters (default: 100)
            - Rule of thumb: sqrt(n_vectors) to 4*sqrt(n_vectors)
            - More clusters = faster search, more memory
        
        n_probe: Clusters to search (default: 10)
            - Higher = better recall, slower search
            - Can be tuned at query time
    
    Complexity:
        - Training: O(n * n_clusters * n_iter)
        - Search: O(n_clusters + (n/n_clusters) * n_probe)
        - Memory: O(n * d + n_clusters * d)
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "euclidean",
        n_clusters: int = 100,
        n_probe: int = 10,
        n_iter: int = 20,
        normalize: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize IVF index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric
            n_clusters: Number of clusters
            n_probe: Number of clusters to search
            n_iter: K-means iterations
            normalize: Whether to L2-normalize vectors
            seed: Random seed
            verbose: Print training progress
        """
        super().__init__(dimension, metric, **kwargs)
        
        self.config = IVFConfig(
            dimension=dimension,
            metric=metric,
            n_clusters=n_clusters,
            n_probe=n_probe,
            n_iter=n_iter,
            normalize=normalize,
            seed=seed,
            verbose=verbose,
        )
        
        # Centroids from k-means
        self._centroids: Optional[NDArray] = None
        
        # Inverted lists: cluster_id -> list of (id, vector, metadata)
        self._inverted_lists: Dict[int, List[VectorEntry]] = defaultdict(list)
        
        # ID to cluster mapping for fast lookup
        self._id_to_cluster: Dict[str, int] = {}
        
        # ID to position in inverted list
        self._id_to_position: Dict[str, int] = {}
        
        # All IDs for iteration
        self._ids: Set[str] = set()
        
        # Distance function
        self._distance_fn = get_metric_fn(metric)
        
        # Statistics
        self._build_start = time.time()
        self._n_distance_computations = 0
        
        # Random state
        self._rng = np.random.RandomState(seed)
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def index_type(self) -> IndexType:
        return IndexType.IVF
    
    @property
    def size(self) -> int:
        return len(self._ids)
    
    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return self.config.n_clusters
    
    @property
    def n_probe(self) -> int:
        """Number of clusters to search."""
        return self.config.n_probe
    
    @property
    def centroids(self) -> Optional[NDArray]:
        """Cluster centroids."""
        return self._centroids
    
    def set_n_probe(self, n_probe: int) -> None:
        """
        Set the number of clusters to search.
        
        Higher values give better recall but slower search.
        
        Args:
            n_probe: Number of clusters to search
        """
        if n_probe < 1:
            raise ValueError(f"n_probe must be >= 1, got {n_probe}")
        self.config.n_probe = min(n_probe, self.config.n_clusters)
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def train(self, vectors: NDArray) -> None:
        """
        Train the index using k-means clustering.
        
        Must be called before adding vectors.
        
        Args:
            vectors: Training vectors (n, dimension)
        """
        vectors = self._validate_vectors(vectors)
        
        if self.config.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            vectors = vectors / norms
        
        n_samples = len(vectors)
        actual_clusters = min(self.config.n_clusters, n_samples)
        
        if self.config.verbose:
            print(f"Training IVF with {actual_clusters} clusters on {n_samples} vectors...")
        
        # Run k-means
        kmeans = KMeans(
            n_clusters=actual_clusters,
            n_iter=self.config.n_iter,
            n_redo=self.config.n_redo,
            seed=self.config.seed,
            verbose=self.config.verbose,
        )
        
        start = time.time()
        self._centroids = kmeans.fit(vectors)
        train_time = time.time() - start
        
        # Update actual cluster count
        self.config.n_clusters = len(self._centroids)
        self.config.n_probe = min(self.config.n_probe, self.config.n_clusters)
        
        # Initialize inverted lists
        self._inverted_lists = {i: [] for i in range(self.config.n_clusters)}
        
        self._is_trained = True
        
        if self.config.verbose:
            print(f"Training completed in {train_time:.2f}s")
    
    def is_trained(self) -> bool:
        """Check if index is trained."""
        return self._is_trained and self._centroids is not None
    
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
            metadata: Optional metadata
            
        Raises:
            RuntimeError: If index is not trained
            ValueError: If ID already exists
        """
        if not self.is_trained():
            raise RuntimeError("Index must be trained before adding vectors")
        
        vector = self._validate_vector(vector)
        
        if self.config.normalize:
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                vector = vector / norm
        
        with self._lock:
            if id in self._ids:
                raise ValueError(f"ID '{id}' already exists in index")
            
            # Find nearest centroid
            cluster_id = self._find_nearest_centroid(vector)
            
            # Add to inverted list
            position = len(self._inverted_lists[cluster_id])
            entry = VectorEntry(id, vector.copy(), metadata or {})
            self._inverted_lists[cluster_id].append(entry)
            
            # Update mappings
            self._id_to_cluster[id] = cluster_id
            self._id_to_position[id] = position
            self._ids.add(id)
    
    def add_batch(
        self,
        ids: List[str],
        vectors: NDArray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add multiple vectors to the index.
        
        Args:
            ids: List of unique identifiers
            vectors: Array of vectors (n, dimension)
            metadata: Optional list of metadata dicts
            
        Returns:
            Number of vectors added
        """
        if not self.is_trained():
            raise RuntimeError("Index must be trained before adding vectors")
        
        vectors = self._validate_vectors(vectors)
        n = len(ids)
        
        if len(vectors) != n:
            raise ValueError(f"Number of ids ({n}) != vectors ({len(vectors)})")
        
        if metadata is not None and len(metadata) != n:
            raise ValueError(f"Number of ids ({n}) != metadata ({len(metadata)})")
        
        if self.config.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            vectors = vectors / norms
        
        with self._lock:
            # Check for duplicates
            for id in ids:
                if id in self._ids:
                    raise ValueError(f"ID '{id}' already exists in index")
            
            # Find nearest centroids for all vectors
            cluster_ids = self._find_nearest_centroids_batch(vectors)
            
            # Add to inverted lists
            for i in range(n):
                cluster_id = cluster_ids[i]
                position = len(self._inverted_lists[cluster_id])
                
                entry = VectorEntry(
                    ids[i],
                    vectors[i].copy(),
                    metadata[i] if metadata else {},
                )
                self._inverted_lists[cluster_id].append(entry)
                
                self._id_to_cluster[ids[i]] = cluster_id
                self._id_to_position[ids[i]] = position
                self._ids.add(ids[i])
        
        return n
    
    def _find_nearest_centroid(self, vector: NDArray) -> int:
        """Find the nearest centroid for a vector."""
        distances = np.array([
            self._distance_fn(vector, c) for c in self._centroids
        ])
        return int(np.argmin(distances))
    
    def _find_nearest_centroids_batch(self, vectors: NDArray) -> NDArray:
        """Find nearest centroids for a batch of vectors."""
        if self._metric == "euclidean":
            distances = pairwise_euclidean(vectors, self._centroids)
        elif self._metric == "cosine":
            distances = pairwise_cosine(vectors, self._centroids)
        else:
            # Fallback
            distances = np.array([
                [self._distance_fn(v, c) for c in self._centroids]
                for v in vectors
            ])
        return np.argmin(distances, axis=1)
    
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
        n_probe: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results
            filter_fn: Optional filter function(id, metadata) -> bool
            include_vectors: Include vectors in results
            include_metadata: Include metadata in results
            n_probe: Override n_probe (optional)
            
        Returns:
            List of SearchResult, sorted by distance
        """
        if not self.is_trained():
            raise RuntimeError("Index must be trained before searching")
        
        if self.size == 0:
            return []
        
        query = self._validate_vector(query)
        
        if self.config.normalize:
            norm = np.linalg.norm(query)
            if norm > 1e-8:
                query = query / norm
        
        n_probe = n_probe or self.config.n_probe
        n_probe = min(n_probe, self.config.n_clusters)
        
        # Find nearest centroids
        centroid_distances = np.array([
            self._distance_fn(query, c) for c in self._centroids
        ])
        nearest_centroids = np.argsort(centroid_distances)[:n_probe]
        
        # Search in selected clusters
        candidates: List[Tuple[float, str, NDArray, Dict]] = []
        
        for cluster_id in nearest_centroids:
            for entry in self._inverted_lists[cluster_id]:
                # Apply filter
                if filter_fn and not filter_fn(entry.id, entry.metadata):
                    continue
                
                dist = self._distance_fn(query, entry.vector)
                self._n_distance_computations += 1
                
                candidates.append((dist, entry.id, entry.vector, entry.metadata))
        
        # Sort and take top k
        candidates.sort(key=lambda x: x[0])
        top_k = candidates[:k]
        
        # Build results
        results = []
        for dist, id, vector, metadata in top_k:
            score = 1.0 / (1.0 + dist)
            
            result = SearchResult(
                id=id,
                distance=dist,
                score=score,
                vector=vector.copy() if include_vectors else None,
                metadata=metadata.copy() if include_metadata else None,
            )
            results.append(result)
        
        return results
    
    def search_batch(
        self,
        queries: NDArray,
        k: int = 10,
        filter_fn: Optional[FilterFunction] = None,
        n_probe: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """
        Search with multiple queries.
        
        Args:
            queries: Array of query vectors (n, dimension)
            k: Number of results per query
            filter_fn: Optional filter function
            n_probe: Override n_probe
            
        Returns:
            List of result lists
        """
        queries = self._validate_vectors(queries)
        return [
            self.search(q, k=k, filter_fn=filter_fn, n_probe=n_probe)
            for q in queries
        ]
    
    def search_with_scores(
        self,
        query: NDArray,
        k: int = 10,
        n_probe: Optional[int] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Search and return IDs with distances.
        
        Simpler interface for benchmarking.
        
        Args:
            query: Query vector
            k: Number of results
            n_probe: Override n_probe
            
        Returns:
            Tuple of (ids, distances)
        """
        results = self.search(
            query, k=k, n_probe=n_probe,
            include_vectors=False, include_metadata=False
        )
        
        ids = [r.id for r in results]
        distances = [r.distance for r in results]
        
        return ids, distances
    
    # =========================================================================
    # REMOVE OPERATIONS
    # =========================================================================
    
    def remove(self, id: str) -> bool:
        """
        Remove a vector from the index.
        
        Note: This marks the vector as deleted but doesn't reclaim space.
        Use rebuild() to fully remove deleted vectors.
        
        Args:
            id: Vector ID to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if id not in self._ids:
                return False
            
            cluster_id = self._id_to_cluster[id]
            position = self._id_to_position[id]
            
            # Remove from inverted list
            # We swap with last element for O(1) removal
            inv_list = self._inverted_lists[cluster_id]
            
            if position < len(inv_list) - 1:
                # Swap with last
                last_entry = inv_list[-1]
                inv_list[position] = last_entry
                self._id_to_position[last_entry.id] = position
            
            inv_list.pop()
            
            # Update mappings
            del self._id_to_cluster[id]
            del self._id_to_position[id]
            self._ids.remove(id)
            
            return True
    
    def remove_batch(self, ids: List[str]) -> int:
        """
        Remove multiple vectors.
        
        Args:
            ids: List of vector IDs
            
        Returns:
            Number removed
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
        """Get a vector by ID."""
        if id not in self._ids:
            return None
        
        cluster_id = self._id_to_cluster[id]
        position = self._id_to_position[id]
        
        entry = self._inverted_lists[cluster_id][position]
        return entry.vector.copy(), entry.metadata.copy()
    
    def contains(self, id: str) -> bool:
        """Check if ID exists."""
        return id in self._ids
    
    def clear(self) -> int:
        """Remove all vectors (keeps centroids)."""
        with self._lock:
            count = len(self._ids)
            
            for cluster_id in self._inverted_lists:
                self._inverted_lists[cluster_id] = []
            
            self._id_to_cluster.clear()
            self._id_to_position.clear()
            self._ids.clear()
            
            return count
    
    def reset(self) -> None:
        """Reset index completely (including centroids)."""
        self.clear()
        self._centroids = None
        self._is_trained = False
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def stats(self) -> IndexStats:
        """Get index statistics."""
        # Cluster statistics
        cluster_sizes = [len(self._inverted_lists[i]) for i in range(self.config.n_clusters)]
        
        # Memory estimation
        vector_memory = self.size * self._dimension * 4  # float32
        centroid_memory = self.config.n_clusters * self._dimension * 4
        metadata_memory = sum(
            len(str(self._inverted_lists[c][i].metadata))
            for c in range(self.config.n_clusters)
            for i in range(len(self._inverted_lists[c]))
        )
        
        return IndexStats(
            index_type=self.index_type.value,
            dimension=self._dimension,
            metric=self._metric,
            vector_count=self.size,
            memory_bytes=vector_memory + centroid_memory + metadata_memory,
            is_trained=self._is_trained,
            build_time_seconds=time.time() - self._build_start,
            extra={
                "n_clusters": self.config.n_clusters,
                "n_probe": self.config.n_probe,
                "cluster_sizes": {
                    "min": min(cluster_sizes) if cluster_sizes else 0,
                    "max": max(cluster_sizes) if cluster_sizes else 0,
                    "mean": np.mean(cluster_sizes) if cluster_sizes else 0,
                    "std": np.std(cluster_sizes) if cluster_sizes else 0,
                },
                "empty_clusters": sum(1 for s in cluster_sizes if s == 0),
                "distance_computations": self._n_distance_computations,
            },
        )
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get detailed cluster information.
        
        Returns:
            Dictionary with cluster statistics
        """
        cluster_sizes = []
        
        for cluster_id in range(self.config.n_clusters):
            size = len(self._inverted_lists[cluster_id])
            cluster_sizes.append({
                "cluster_id": cluster_id,
                "size": size,
            })
        
        # Sort by size
        cluster_sizes.sort(key=lambda x: x["size"], reverse=True)
        
        sizes = [c["size"] for c in cluster_sizes]
        
        return {
            "n_clusters": self.config.n_clusters,
            "total_vectors": self.size,
            "cluster_sizes": cluster_sizes,
            "size_distribution": {
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "mean": np.mean(sizes) if sizes else 0,
                "std": np.std(sizes) if sizes else 0,
                "median": np.median(sizes) if sizes else 0,
            },
            "empty_clusters": sum(1 for s in sizes if s == 0),
            "imbalance_ratio": max(sizes) / (np.mean(sizes) + 1e-8) if sizes else 0,
        }
    
    def get_vectors_in_cluster(self, cluster_id: int) -> List[str]:
        """
        Get all vector IDs in a cluster.
        
        Args:
            cluster_id: Cluster index
            
        Returns:
            List of vector IDs
        """
        if cluster_id < 0 or cluster_id >= self.config.n_clusters:
            raise ValueError(f"Invalid cluster_id: {cluster_id}")
        
        return [entry.id for entry in self._inverted_lists[cluster_id]]
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize index to dictionary."""
        inverted_lists_data = {}
        
        for cluster_id, entries in self._inverted_lists.items():
            inverted_lists_data[str(cluster_id)] = [
                {
                    "id": entry.id,
                    "vector": entry.vector.tolist(),
                    "metadata": entry.metadata,
                }
                for entry in entries
            ]
        
        return {
            "index_type": self.index_type.value,
            "dimension": self._dimension,
            "metric": self._metric,
            "config": {
                "n_clusters": self.config.n_clusters,
                "n_probe": self.config.n_probe,
                "n_iter": self.config.n_iter,
                "normalize": self.config.normalize,
                "seed": self.config.seed,
            },
            "is_trained": self._is_trained,
            "centroids": self._centroids.tolist() if self._centroids is not None else None,
            "inverted_lists": inverted_lists_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IVFIndex":
        """Deserialize index from dictionary."""
        config = data.get("config", {})
        
        index = cls(
            dimension=data["dimension"],
            metric=data["metric"],
            n_clusters=config.get("n_clusters", 100),
            n_probe=config.get("n_probe", 10),
            n_iter=config.get("n_iter", 20),
            normalize=config.get("normalize", False),
            seed=config.get("seed"),
        )
        
        # Restore training state
        if data.get("centroids") is not None:
            index._centroids = np.array(data["centroids"], dtype=np.float32)
            index._is_trained = data.get("is_trained", True)
            
            # Initialize inverted lists
            index._inverted_lists = {
                i: [] for i in range(index.config.n_clusters)
            }
        
        # Restore vectors
        inverted_lists_data = data.get("inverted_lists", {})
        
        for cluster_id_str, entries_data in inverted_lists_data.items():
            cluster_id = int(cluster_id_str)
            
            for entry_data in entries_data:
                entry = VectorEntry(
                    id=entry_data["id"],
                    vector=np.array(entry_data["vector"], dtype=np.float32),
                    metadata=entry_data.get("metadata", {}),
                )
                
                position = len(index._inverted_lists[cluster_id])
                index._inverted_lists[cluster_id].append(entry)
                
                index._id_to_cluster[entry.id] = cluster_id
                index._id_to_position[entry.id] = position
                index._ids.add(entry.id)
        
        return index
    
    # =========================================================================
    # ITERATION
    # =========================================================================
    
    def iter_ids(self) -> Iterator[str]:
        """Iterate over all IDs."""
        return iter(self._ids)
    
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        """Iterate over all vectors."""
        for cluster_id in range(self.config.n_clusters):
            for entry in self._inverted_lists[cluster_id]:
                yield entry.id, entry.vector.copy(), entry.metadata.copy()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def rebuild(self, vectors: Optional[NDArray] = None) -> None:
        """
        Rebuild the index.
        
        Re-trains centroids and re-assigns all vectors.
        Useful after many deletions or to improve cluster balance.
        
        Args:
            vectors: Optional new training vectors (uses existing if None)
        """
        if self.size == 0 and vectors is None:
            return
        
        # Collect all current vectors
        all_ids = []
        all_vectors = []
        all_metadata = []
        
        for id, vector, metadata in self.iter_vectors():
            all_ids.append(id)
            all_vectors.append(vector)
            all_metadata.append(metadata)
        
        all_vectors = np.array(all_vectors, dtype=np.float32) if all_vectors else None
        
        # Reset
        self.reset()
        
        # Re-train
        training_vectors = vectors if vectors is not None else all_vectors
        if training_vectors is not None and len(training_vectors) > 0:
            self.train(training_vectors)
        
        # Re-add vectors
        if all_vectors is not None and len(all_vectors) > 0:
            self.add_batch(all_ids, all_vectors, all_metadata)
    
    def assign_cluster(self, vector: NDArray) -> int:
        """
        Get the cluster assignment for a vector.
        
        Args:
            vector: Vector to assign
            
        Returns:
            Cluster ID
        """
        if not self.is_trained():
            raise RuntimeError("Index must be trained")
        
        vector = self._validate_vector(vector)
        return self._find_nearest_centroid(vector)
    
    def get_centroid(self, cluster_id: int) -> NDArray:
        """
        Get a cluster centroid.
        
        Args:
            cluster_id: Cluster index
            
        Returns:
            Centroid vector
        """
        if not self.is_trained():
            raise RuntimeError("Index must be trained")
        
        if cluster_id < 0 or cluster_id >= self.config.n_clusters:
            raise ValueError(f"Invalid cluster_id: {cluster_id}")
        
        return self._centroids[cluster_id].copy()