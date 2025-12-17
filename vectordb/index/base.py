"""
Abstract base class for all index implementations.

This module defines the interface that all indices must implement,
ensuring consistent behavior across different index types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    Callable,
    Iterator,
    Union,
    Set,
)
import numpy as np
from numpy.typing import NDArray
import threading
import time


class IndexType(str, Enum):
    """Available index types."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    PQ = "pq"
    IVF_PQ = "ivf_pq"


@dataclass
class IndexConfig:
    """Base configuration for indices."""
    
    dimension: int
    metric: str = "euclidean"
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {self.dimension}")


@dataclass
class IndexStats:
    """Statistics about an index."""
    
    index_type: str
    dimension: int
    metric: str
    vector_count: int
    memory_bytes: int
    is_trained: bool
    build_time_seconds: float = 0.0
    
    # Optional type-specific stats
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "metric": self.metric,
            "vector_count": self.vector_count,
            "memory_bytes": self.memory_bytes,
            "memory_mb": round(self.memory_bytes / (1024 * 1024), 2),
            "is_trained": self.is_trained,
            "build_time_seconds": self.build_time_seconds,
            **self.extra,
        }


@dataclass
class SearchResult:
    """
    Result from an index search.
    
    Attributes:
        id: Vector ID
        distance: Distance from query
        score: Similarity score (higher = more similar)
        vector: The vector (optional)
        metadata: Associated metadata (optional)
    """
    
    id: str
    distance: float
    score: float = 0.0
    vector: Optional[NDArray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Compute score from distance if not set
        if self.score == 0.0 and self.distance >= 0:
            # Simple inverse distance score
            self.score = 1.0 / (1.0 + self.distance)
    
    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "distance": self.distance,
            "score": self.score,
        }
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if include_vector and self.vector is not None:
            result["vector"] = self.vector.tolist()
        return result
    
    def __repr__(self) -> str:
        return f"SearchResult(id='{self.id}', distance={self.distance:.4f})"


# Type alias for filter function
FilterFunction = Callable[[str, Dict[str, Any]], bool]


class BaseIndex(ABC):
    """
    Abstract base class for vector indices.
    
    All index implementations must inherit from this class and
    implement the required abstract methods.
    
    Thread Safety:
        All implementations should be thread-safe for concurrent
        reads. Write operations may require external synchronization.
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "euclidean",
        **kwargs,
    ):
        """
        Initialize index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric
            **kwargs: Index-specific parameters
        """
        self._dimension = dimension
        self._metric = metric
        self._is_trained = False
        self._build_time = 0.0
        self._lock = threading.RLock()
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def dimension(self) -> int:
        """Vector dimension."""
        return self._dimension
    
    @property
    def metric(self) -> str:
        """Distance metric."""
        return self._metric
    
    @property
    def is_trained(self) -> bool:
        """Whether index is trained (for indices that require training)."""
        return self._is_trained
    
    @property
    @abstractmethod
    def index_type(self) -> IndexType:
        """Return the index type."""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors in the index."""
        pass
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
            List of SearchResult, sorted by distance
        """
        pass
    
    @abstractmethod
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
            List of result lists
        """
        pass
    
    @abstractmethod
    def remove(self, id: str) -> bool:
        """
        Remove a vector from the index.
        
        Args:
            id: Vector ID to remove
            
        Returns:
            True if removed, False if not found
        """
        pass
    
    @abstractmethod
    def remove_batch(self, ids: List[str]) -> int:
        """
        Remove multiple vectors.
        
        Args:
            ids: List of vector IDs to remove
            
        Returns:
            Number of vectors removed
        """
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[Tuple[NDArray, Dict[str, Any]]]:
        """
        Get a vector by ID.
        
        Args:
            id: Vector ID
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        pass
    
    @abstractmethod
    def contains(self, id: str) -> bool:
        """
        Check if vector ID exists.
        
        Args:
            id: Vector ID
            
        Returns:
            True if exists
        """
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """
        Remove all vectors from the index.
        
        Returns:
            Number of vectors removed
        """
        pass
    
    @abstractmethod
    def stats(self) -> IndexStats:
        """
        Get index statistics.
        
        Returns:
            IndexStats object
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS (with default implementations)
    # =========================================================================
    
    def train(self, vectors: NDArray) -> None:
        """
        Train the index (for indices that require training).
        
        Default implementation does nothing.
        
        Args:
            vectors: Training vectors
        """
        self._is_trained = True
    
    def rebuild(self) -> None:
        """
        Rebuild the index.
        
        Default implementation does nothing.
        """
        pass
    
    def optimize(self) -> None:
        """
        Optimize the index for search performance.
        
        Default implementation does nothing.
        """
        pass
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize index to dictionary.
        
        Returns:
            Dictionary representation
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseIndex":
        """
        Deserialize index from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Index instance
        """
        pass
    
    # =========================================================================
    # ITERATION
    # =========================================================================
    
    @abstractmethod
    def iter_ids(self) -> Iterator[str]:
        """
        Iterate over all vector IDs.
        
        Returns:
            Iterator of IDs
        """
        pass
    
    @abstractmethod
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        """
        Iterate over all vectors.
        
        Returns:
            Iterator of (id, vector, metadata) tuples
        """
        pass
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _validate_vector(self, vector: NDArray) -> NDArray:
        """Validate and normalize vector."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        
        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1D, got {vector.ndim}D")
        
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} != index dimension {self._dimension}"
            )
        
        return vector
    
    def _validate_vectors(self, vectors: NDArray) -> NDArray:
        """Validate and normalize batch of vectors."""
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D, got {vectors.ndim}D")
        
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != index dimension {self._dimension}"
            )
        
        return vectors
    
    def __len__(self) -> int:
        """Number of vectors."""
        return self.size
    
    def __contains__(self, id: str) -> bool:
        """Check if ID exists."""
        return self.contains(id)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dimension={self._dimension}, "
            f"metric='{self._metric}', "
            f"size={self.size})"
        )