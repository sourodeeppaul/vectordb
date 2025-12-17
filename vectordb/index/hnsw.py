"""
HNSW (Hierarchical Navigable Small World) Index Implementation.

HNSW is a graph-based approximate nearest neighbor algorithm that
provides logarithmic search complexity with high recall.

Key Features:
    - O(log n) search complexity
    - High recall (typically 95-99%)
    - Efficient for high-dimensional data
    - Supports incremental insertions

Reference:
    Malkov, Y. A., & Yashunin, D. A. (2018).
    "Efficient and robust approximate nearest neighbor search using
    Hierarchical Navigable Small World graphs."
    https://arxiv.org/abs/1603.09320
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import threading
import time
import heapq
import math
import random
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
from ..distance import get_metric_fn, query_distances


class Neighbor(NamedTuple):
    """A neighbor with its distance."""
    distance: float
    id: str


@dataclass
class HNSWConfig(IndexConfig):
    """Configuration for HNSW index."""
    
    # Maximum number of connections per node (except layer 0)
    M: int = 16
    
    # Maximum connections at layer 0 (typically 2*M)
    M_max0: int = 32
    
    # Size of dynamic candidate list during construction
    ef_construction: int = 200
    
    # Size of dynamic candidate list during search
    ef_search: int = 50
    
    # Level generation multiplier (1/ln(M))
    ml: float = None
    
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    # Whether to extend candidates during search
    extend_candidates: bool = True
    
    # Whether to keep pruned connections
    keep_pruned_connections: bool = True
    
    # Normalize vectors
    normalize: bool = False
    
    def __post_init__(self):
        super().validate()
        
        if self.M_max0 is None:
            self.M_max0 = self.M * 2
        
        if self.ml is None:
            self.ml = 1.0 / math.log(self.M)


class HNSWNode:
    """
    A node in the HNSW graph.
    
    Each node stores:
    - The vector data
    - Connections at each layer
    - Metadata
    """
    
    __slots__ = ['id', 'vector', 'metadata', 'layer', 'neighbors']
    
    def __init__(
        self,
        id: str,
        vector: NDArray,
        layer: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.vector = vector
        self.metadata = metadata or {}
        self.layer = layer  # Maximum layer this node exists in
        # neighbors[level] = list of neighbor IDs
        self.neighbors: Dict[int, List[str]] = {l: [] for l in range(layer + 1)}
    
    def get_neighbors(self, level: int) -> List[str]:
        """Get neighbors at a specific level."""
        return self.neighbors.get(level, [])
    
    def add_neighbor(self, level: int, neighbor_id: str) -> None:
        """Add a neighbor at a specific level."""
        if level not in self.neighbors:
            self.neighbors[level] = []
        if neighbor_id not in self.neighbors[level]:
            self.neighbors[level].append(neighbor_id)
    
    def remove_neighbor(self, level: int, neighbor_id: str) -> None:
        """Remove a neighbor at a specific level."""
        if level in self.neighbors and neighbor_id in self.neighbors[level]:
            self.neighbors[level].remove(neighbor_id)
    
    def set_neighbors(self, level: int, neighbors: List[str]) -> None:
        """Set all neighbors at a specific level."""
        self.neighbors[level] = neighbors


class HNSWIndex(BaseIndex):
    """
    HNSW (Hierarchical Navigable Small World) Index.
    
    A graph-based index providing fast approximate nearest neighbor search
    with high recall and logarithmic complexity.
    
    Example:
        >>> index = HNSWIndex(dimension=128, metric="cosine")
        >>> 
        >>> # Add vectors
        >>> for i in range(10000):
        ...     index.add(f"vec{i}", vectors[i], {"idx": i})
        >>> 
        >>> # Search (fast!)
        >>> results = index.search(query, k=10)
        >>> 
        >>> # Tune search quality vs speed
        >>> index.set_ef_search(100)  # Higher = better recall, slower
        >>> results = index.search(query, k=10)
    
    Parameters:
        M: Max connections per node (default: 16)
            - Higher M = better recall, more memory, slower construction
            - Typical values: 12-48
        
        ef_construction: Construction beam width (default: 200)
            - Higher = better graph quality, slower construction
            - Should be >= M
        
        ef_search: Search beam width (default: 50)
            - Higher = better recall, slower search
            - Can be tuned at query time
    
    Complexity:
        - Construction: O(n * log(n) * M * ef_construction)
        - Search: O(log(n) * ef_search)
        - Memory: O(n * M)
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "euclidean",
        M: int = 16,
        M_max0: int = None,
        ef_construction: int = 200,
        ef_search: int = 50,
        normalize: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize HNSW index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric
            M: Max connections per node (except layer 0)
            M_max0: Max connections at layer 0 (default: 2*M)
            ef_construction: Beam width during construction
            ef_search: Beam width during search
            normalize: Whether to L2-normalize vectors
            seed: Random seed for reproducibility
        """
        super().__init__(dimension, metric, **kwargs)
        
        if M_max0 is None:
            M_max0 = M * 2
        
        self.config = HNSWConfig(
            dimension=dimension,
            metric=metric,
            M=M,
            M_max0=M_max0,
            ef_construction=ef_construction,
            ef_search=ef_search,
            normalize=normalize,
            seed=seed,
        )
        
        # Graph structure
        self._nodes: Dict[str, HNSWNode] = {}
        self._entry_point: Optional[str] = None
        self._max_level: int = -1
        
        # Random number generator
        self._rng = random.Random(seed)
        
        # Distance function
        self._distance_fn = get_metric_fn(metric)
        
        # Statistics
        self._build_start = time.time()
        self._n_distance_computations = 0
        
        # Already trained (no training needed for HNSW)
        self._is_trained = True
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def index_type(self) -> IndexType:
        return IndexType.HNSW
    
    @property
    def size(self) -> int:
        return len(self._nodes)
    
    @property
    def entry_point(self) -> Optional[str]:
        """Current entry point ID."""
        return self._entry_point
    
    @property
    def max_level(self) -> int:
        """Current maximum level in the graph."""
        return self._max_level
    
    def set_ef_search(self, ef: int) -> None:
        """
        Set the search beam width.
        
        Higher values give better recall but slower search.
        
        Args:
            ef: New ef_search value (should be >= k for search)
        """
        self.config.ef_search = ef
    
    # =========================================================================
    # CORE ALGORITHMS
    # =========================================================================
    
    def _random_level(self) -> int:
        """
        Generate a random level for a new node.
        
        Uses exponential distribution: P(level=l) = (1/M)^l * (1 - 1/M)
        """
        level = 0
        while self._rng.random() < math.exp(-level * self.config.ml):
            level += 1
            if level > 50:  # Safety cap
                break
        return level
    
    def _distance(self, id1: str, id2: str) -> float:
        """Compute distance between two nodes."""
        self._n_distance_computations += 1
        return self._distance_fn(
            self._nodes[id1].vector,
            self._nodes[id2].vector
        )
    
    def _distance_to_query(self, query: NDArray, id: str) -> float:
        """Compute distance from query to a node."""
        self._n_distance_computations += 1
        return self._distance_fn(query, self._nodes[id].vector)
    
    def _search_layer(
        self,
        query: NDArray,
        entry_points: List[str],
        ef: int,
        level: int,
    ) -> List[Neighbor]:
        """
        Search a single layer of the graph.
        
        Implements Algorithm 2 from the HNSW paper.
        
        Args:
            query: Query vector
            entry_points: Starting points for search
            ef: Number of candidates to track
            level: Layer to search
            
        Returns:
            List of (distance, id) sorted by distance
        """
        # Visited set
        visited: Set[str] = set(entry_points)
        
        # Candidates (min-heap by distance)
        candidates: List[Tuple[float, str]] = []
        
        # Results (max-heap by negative distance to get furthest)
        results: List[Tuple[float, str]] = []
        
        # Initialize with entry points
        for ep in entry_points:
            dist = self._distance_to_query(query, ep)
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))
        
        # Search
        while candidates:
            # Get closest candidate
            dist_c, current = heapq.heappop(candidates)
            
            # Get furthest result
            dist_f = -results[0][0]
            
            # Stop if closest candidate is further than furthest result
            if dist_c > dist_f:
                break
            
            # Explore neighbors
            node = self._nodes[current]
            for neighbor_id in node.get_neighbors(level):
                if neighbor_id in visited:
                    continue
                
                visited.add(neighbor_id)
                
                dist_n = self._distance_to_query(query, neighbor_id)
                dist_f = -results[0][0]
                
                # Add to candidates if better than worst result or results not full
                if dist_n < dist_f or len(results) < ef:
                    heapq.heappush(candidates, (dist_n, neighbor_id))
                    heapq.heappush(results, (-dist_n, neighbor_id))
                    
                    # Trim results to ef
                    if len(results) > ef:
                        heapq.heappop(results)
        
        # Convert to sorted list
        return sorted([Neighbor(-d, id) for d, id in results])
    
    def _select_neighbors_simple(
        self,
        query_id: str,
        candidates: List[Neighbor],
        M: int,
    ) -> List[str]:
        """
        Simple neighbor selection (Algorithm 3).
        
        Just returns the M closest candidates.
        """
        return [n.id for n in candidates[:M]]
    
    def _select_neighbors_heuristic(
        self,
        query_id: str,
        candidates: List[Neighbor],
        M: int,
        level: int,
        extend_candidates: bool = True,
        keep_pruned: bool = True,
    ) -> List[str]:
        """
        Heuristic neighbor selection (Algorithm 4).
        
        Selects diverse neighbors that provide good graph connectivity.
        """
        if len(candidates) <= M:
            return [n.id for n in candidates]
        
        # Extend candidates with their neighbors
        if extend_candidates:
            extended: Dict[str, float] = {n.id: n.distance for n in candidates}
            
            for neighbor in candidates:
                if neighbor.id not in self._nodes:
                    continue
                node = self._nodes[neighbor.id]
                for nn_id in node.get_neighbors(level):
                    if nn_id not in extended and nn_id != query_id:
                        dist = self._distance(query_id, nn_id)
                        extended[nn_id] = dist
            
            candidates = sorted([
                Neighbor(d, id) for id, d in extended.items()
            ])
        
        # Select neighbors using heuristic
        selected: List[str] = []
        pruned: List[Neighbor] = []
        
        for candidate in candidates:
            if len(selected) >= M:
                break
            
            # Check if candidate is closer to query than to any selected neighbor
            is_good = True
            for sel_id in selected:
                dist_to_selected = self._distance(candidate.id, sel_id)
                if dist_to_selected < candidate.distance:
                    is_good = False
                    pruned.append(candidate)
                    break
            
            if is_good:
                selected.append(candidate.id)
        
        # Add pruned connections if not enough
        if keep_pruned and len(selected) < M:
            for p in pruned:
                if len(selected) >= M:
                    break
                if p.id not in selected:
                    selected.append(p.id)
        
        return selected
    
    def _connect_neighbors(
        self,
        node_id: str,
        neighbors: List[str],
        level: int,
    ) -> None:
        """
        Connect a node to its neighbors (bidirectional).
        
        Also handles pruning if neighbors exceed M.
        """
        node = self._nodes[node_id]
        M_max = self.config.M_max0 if level == 0 else self.config.M
        
        # Set forward connections
        node.set_neighbors(level, neighbors[:M_max])
        
        # Add back connections
        for neighbor_id in neighbors:
            if neighbor_id not in self._nodes:
                continue
            
            neighbor = self._nodes[neighbor_id]
            neighbor_neighbors = neighbor.get_neighbors(level)
            
            if node_id not in neighbor_neighbors:
                if len(neighbor_neighbors) < M_max:
                    neighbor.add_neighbor(level, node_id)
                else:
                    # Need to prune
                    candidates = [
                        Neighbor(self._distance(neighbor_id, nn), nn)
                        for nn in neighbor_neighbors
                    ]
                    candidates.append(
                        Neighbor(self._distance(neighbor_id, node_id), node_id)
                    )
                    candidates.sort()
                    
                    new_neighbors = self._select_neighbors_heuristic(
                        neighbor_id, candidates, M_max, level,
                        extend_candidates=False, keep_pruned=True
                    )
                    neighbor.set_neighbors(level, new_neighbors)
    
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
        
        Implements Algorithm 1 from the HNSW paper.
        
        Args:
            id: Unique identifier
            vector: Vector to add
            metadata: Optional metadata
        """
        vector = self._validate_vector(vector)
        
        if self.config.normalize:
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                vector = vector / norm
        
        with self._lock:
            if id in self._nodes:
                raise ValueError(f"ID '{id}' already exists in index")
            
            # Generate random level for new node
            level = self._random_level()
            
            # Create node
            node = HNSWNode(id, vector.copy(), level, metadata)
            self._nodes[id] = node
            
            # First node
            if self._entry_point is None:
                self._entry_point = id
                self._max_level = level
                return
            
            # Find entry point for insertion
            current_ep = self._entry_point
            
            # Traverse from top to node's level + 1
            for lc in range(self._max_level, level, -1):
                neighbors = self._search_layer(vector, [current_ep], 1, lc)
                if neighbors:
                    current_ep = neighbors[0].id
            
            # Insert at each level from level down to 0
            for lc in range(min(level, self._max_level), -1, -1):
                # Search for neighbors
                neighbors = self._search_layer(
                    vector, [current_ep], self.config.ef_construction, lc
                )
                
                # Select neighbors
                M = self.config.M_max0 if lc == 0 else self.config.M
                selected = self._select_neighbors_heuristic(
                    id, neighbors, M, lc,
                    extend_candidates=self.config.extend_candidates,
                    keep_pruned=self.config.keep_pruned_connections,
                )
                
                # Connect
                self._connect_neighbors(id, selected, lc)
                
                # Update entry point for next level
                if neighbors:
                    current_ep = neighbors[0].id
            
            # Update entry point if new node has higher level
            if level > self._max_level:
                self._entry_point = id
                self._max_level = level
    
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
        vectors = self._validate_vectors(vectors)
        n = len(ids)
        
        if len(vectors) != n:
            raise ValueError(f"Number of ids ({n}) != vectors ({len(vectors)})")
        
        if metadata is not None and len(metadata) != n:
            raise ValueError(f"Number of ids ({n}) != metadata ({len(metadata)})")
        
        for i in range(n):
            self.add(
                ids[i],
                vectors[i],
                metadata[i] if metadata else None
            )
        
        return n
    
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
        ef: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Implements Algorithm 5 from the HNSW paper.
        
        Args:
            query: Query vector
            k: Number of results
            filter_fn: Optional filter function(id, metadata) -> bool
            include_vectors: Include vectors in results
            include_metadata: Include metadata in results
            ef: Override ef_search (optional)
            
        Returns:
            List of SearchResult, sorted by distance
        """
        if self.size == 0:
            return []
        
        query = self._validate_vector(query)
        
        if self.config.normalize:
            norm = np.linalg.norm(query)
            if norm > 1e-8:
                query = query / norm
        
        ef = ef or self.config.ef_search
        ef = max(ef, k)  # ef should be at least k
        
        # Start from entry point
        current_ep = self._entry_point
        
        # Traverse from top level to level 1
        for level in range(self._max_level, 0, -1):
            neighbors = self._search_layer(query, [current_ep], 1, level)
            if neighbors:
                current_ep = neighbors[0].id
        
        # Search at level 0 with ef
        candidates = self._search_layer(query, [current_ep], ef, 0)
        
        # Apply filter and build results
        results = []
        for neighbor in candidates:
            if len(results) >= k:
                break
            
            node = self._nodes[neighbor.id]
            
            # Apply filter
            if filter_fn and not filter_fn(neighbor.id, node.metadata):
                continue
            
            # Compute score
            score = 1.0 / (1.0 + neighbor.distance)
            
            result = SearchResult(
                id=neighbor.id,
                distance=neighbor.distance,
                score=score,
                vector=node.vector.copy() if include_vectors else None,
                metadata=node.metadata.copy() if include_metadata else None,
            )
            results.append(result)
        
        return results
    
    def search_batch(
        self,
        queries: NDArray,
        k: int = 10,
        filter_fn: Optional[FilterFunction] = None,
        ef: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """
        Search with multiple queries.
        
        Args:
            queries: Array of query vectors (n, dimension)
            k: Number of results per query
            filter_fn: Optional filter function
            ef: Override ef_search
            
        Returns:
            List of result lists
        """
        queries = self._validate_vectors(queries)
        return [
            self.search(q, k=k, filter_fn=filter_fn, ef=ef)
            for q in queries
        ]
    
    # =========================================================================
    # REMOVE OPERATIONS
    # =========================================================================
    
    def remove(self, id: str) -> bool:
        """
        Remove a vector from the index.
        
        Note: Removal in HNSW can affect graph quality.
        For frequent removals, consider periodic rebuilding.
        
        Args:
            id: Vector ID to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if id not in self._nodes:
                return False
            
            node = self._nodes[id]
            
            # Remove connections to this node from all neighbors
            for level in range(node.layer + 1):
                for neighbor_id in node.get_neighbors(level):
                    if neighbor_id in self._nodes:
                        self._nodes[neighbor_id].remove_neighbor(level, id)
            
            # If this was entry point, find new one
            if id == self._entry_point:
                if len(self._nodes) == 1:
                    self._entry_point = None
                    self._max_level = -1
                else:
                    # Find node at highest level
                    best_id = None
                    best_level = -1
                    for nid, n in self._nodes.items():
                        if nid != id and n.layer > best_level:
                            best_id = nid
                            best_level = n.layer
                    self._entry_point = best_id
                    self._max_level = best_level
            
            del self._nodes[id]
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
        if id not in self._nodes:
            return None
        
        node = self._nodes[id]
        return node.vector.copy(), node.metadata.copy()
    
    def contains(self, id: str) -> bool:
        """Check if ID exists."""
        return id in self._nodes
    
    def clear(self) -> int:
        """Remove all vectors."""
        with self._lock:
            count = len(self._nodes)
            self._nodes.clear()
            self._entry_point = None
            self._max_level = -1
            return count
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def stats(self) -> IndexStats:
        """Get index statistics."""
        # Count nodes at each level
        level_counts = defaultdict(int)
        total_connections = 0
        
        for node in self._nodes.values():
            level_counts[node.layer] += 1
            for level in range(node.layer + 1):
                total_connections += len(node.get_neighbors(level))
        
        # Memory estimation
        vector_memory = sum(n.vector.nbytes for n in self._nodes.values())
        # Rough estimate for graph structure
        graph_memory = total_connections * 50  # ~50 bytes per connection
        metadata_memory = sum(
            len(str(n.metadata)) for n in self._nodes.values()
        )
        
        return IndexStats(
            index_type=self.index_type.value,
            dimension=self._dimension,
            metric=self._metric,
            vector_count=self.size,
            memory_bytes=vector_memory + graph_memory + metadata_memory,
            is_trained=True,
            build_time_seconds=time.time() - self._build_start,
            extra={
                "M": self.config.M,
                "M_max0": self.config.M_max0,
                "ef_construction": self.config.ef_construction,
                "ef_search": self.config.ef_search,
                "max_level": self._max_level,
                "entry_point": self._entry_point,
                "level_distribution": dict(level_counts),
                "total_connections": total_connections,
                "avg_connections": total_connections / max(1, self.size),
                "distance_computations": self._n_distance_computations,
            },
        )
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get detailed graph information.
        
        Returns:
            Dictionary with graph structure info
        """
        level_stats = []
        
        for level in range(self._max_level + 1):
            nodes_at_level = [
                n for n in self._nodes.values()
                if n.layer >= level
            ]
            
            if not nodes_at_level:
                continue
            
            connection_counts = [
                len(n.get_neighbors(level)) for n in nodes_at_level
            ]
            
            level_stats.append({
                "level": level,
                "node_count": len(nodes_at_level),
                "avg_connections": np.mean(connection_counts),
                "min_connections": min(connection_counts),
                "max_connections": max(connection_counts),
            })
        
        return {
            "max_level": self._max_level,
            "entry_point": self._entry_point,
            "total_nodes": self.size,
            "levels": level_stats,
        }
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize index to dictionary."""
        nodes_data = {}
        
        for id, node in self._nodes.items():
            nodes_data[id] = {
                "vector": node.vector.tolist(),
                "metadata": node.metadata,
                "layer": node.layer,
                "neighbors": {
                    str(level): neighbors
                    for level, neighbors in node.neighbors.items()
                },
            }
        
        return {
            "index_type": self.index_type.value,
            "dimension": self._dimension,
            "metric": self._metric,
            "config": {
                "M": self.config.M,
                "M_max0": self.config.M_max0,
                "ef_construction": self.config.ef_construction,
                "ef_search": self.config.ef_search,
                "normalize": self.config.normalize,
                "seed": self.config.seed,
            },
            "entry_point": self._entry_point,
            "max_level": self._max_level,
            "nodes": nodes_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HNSWIndex":
        """Deserialize index from dictionary."""
        config = data.get("config", {})
        
        index = cls(
            dimension=data["dimension"],
            metric=data["metric"],
            M=config.get("M", 16),
            M_max0=config.get("M_max0"),
            ef_construction=config.get("ef_construction", 200),
            ef_search=config.get("ef_search", 50),
            normalize=config.get("normalize", False),
            seed=config.get("seed"),
        )
        
        # Restore graph structure
        index._entry_point = data.get("entry_point")
        index._max_level = data.get("max_level", -1)
        
        # Restore nodes
        for id, node_data in data.get("nodes", {}).items():
            node = HNSWNode(
                id=id,
                vector=np.array(node_data["vector"], dtype=np.float32),
                layer=node_data["layer"],
                metadata=node_data.get("metadata", {}),
            )
            
            # Restore neighbors
            for level_str, neighbors in node_data.get("neighbors", {}).items():
                node.neighbors[int(level_str)] = neighbors
            
            index._nodes[id] = node
        
        return index
    
    # =========================================================================
    # ITERATION
    # =========================================================================
    
    def iter_ids(self) -> Iterator[str]:
        """Iterate over all IDs."""
        return iter(self._nodes.keys())
    
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        """Iterate over all vectors."""
        for id, node in self._nodes.items():
            yield id, node.vector.copy(), node.metadata.copy()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_neighbors(self, id: str, level: int = 0) -> List[str]:
        """
        Get neighbors of a node at a specific level.
        
        Args:
            id: Node ID
            level: Level (default: 0)
            
        Returns:
            List of neighbor IDs
        """
        if id not in self._nodes:
            return []
        return self._nodes[id].get_neighbors(level).copy()
    
    def get_node_level(self, id: str) -> int:
        """
        Get the maximum level of a node.
        
        Args:
            id: Node ID
            
        Returns:
            Maximum level (-1 if not found)
        """
        if id not in self._nodes:
            return -1
        return self._nodes[id].layer
    
    def rebuild(self) -> None:
        """
        Rebuild the index from scratch.
        
        Useful after many deletions to restore graph quality.
        """
        if self.size == 0:
            return
        
        # Save all vectors
        all_data = [
            (id, node.vector.copy(), node.metadata.copy())
            for id, node in self._nodes.items()
        ]
        
        # Clear and rebuild
        self._nodes.clear()
        self._entry_point = None
        self._max_level = -1
        
        for id, vector, metadata in all_data:
            self.add(id, vector, metadata)