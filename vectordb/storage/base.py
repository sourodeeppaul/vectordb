"""
Abstract base class for storage backends.
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
    Iterator,
    Set,
)
import numpy as np
from numpy.typing import NDArray
import threading


class StorageMode(str, Enum):
    """Storage access modes."""
    READ_ONLY = "r"
    READ_WRITE = "rw"
    CREATE = "c"
    TRUNCATE = "w"


@dataclass
class StorageConfig:
    """Configuration for storage backends."""
    
    dimension: int
    mode: StorageMode = StorageMode.READ_WRITE
    
    # Memory options
    max_memory_mb: Optional[int] = None
    
    # Disk options
    sync_on_write: bool = False
    use_mmap: bool = True
    compression: Optional[str] = None  # None, "lz4", "zstd"
    
    # Cache options
    cache_size_mb: int = 64


@dataclass
class StorageStats:
    """Statistics about storage."""
    
    vector_count: int
    dimension: int
    memory_bytes: int
    disk_bytes: int = 0
    
    # Performance stats
    reads: int = 0
    writes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "memory_mb": round(self.memory_bytes / (1024 * 1024), 2),
            "disk_mb": round(self.disk_bytes / (1024 * 1024), 2),
            "reads": self.reads,
            "writes": self.writes,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
        }


class BaseStorage(ABC):
    """
    Abstract base class for storage backends.
    
    All storage implementations must provide:
    - Vector CRUD operations
    - Batch operations
    - Iteration
    - Persistence (if applicable)
    """
    
    def __init__(self, dimension: int, **kwargs):
        self._dimension = dimension
        self._lock = threading.RLock()
    
    @property
    def dimension(self) -> int:
        """Vector dimension."""
        return self._dimension
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors stored."""
        pass
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    @abstractmethod
    def put(
        self,
        id: str,
        vector: NDArray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a vector.
        
        Args:
            id: Unique identifier
            vector: Vector data
            metadata: Optional metadata
        """
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[Tuple[NDArray, Dict[str, Any]]]:
        """
        Retrieve a vector.
        
        Args:
            id: Vector ID
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """
        Delete a vector.
        
        Args:
            id: Vector ID
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def contains(self, id: str) -> bool:
        """Check if ID exists."""
        pass
    
    @abstractmethod
    def update(
        self,
        id: str,
        vector: Optional[NDArray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update a vector.
        
        Args:
            id: Vector ID
            vector: New vector (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if updated, False if not found
        """
        pass
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    @abstractmethod
    def put_batch(
        self,
        ids: List[str],
        vectors: NDArray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Store multiple vectors.
        
        Returns:
            Number of vectors stored
        """
        pass
    
    @abstractmethod
    def get_batch(
        self,
        ids: List[str]
    ) -> List[Optional[Tuple[NDArray, Dict[str, Any]]]]:
        """
        Retrieve multiple vectors.
        
        Returns:
            List of (vector, metadata) or None for each ID
        """
        pass
    
    @abstractmethod
    def delete_batch(self, ids: List[str]) -> int:
        """
        Delete multiple vectors.
        
        Returns:
            Number of vectors deleted
        """
        pass
    
    # =========================================================================
    # BULK ACCESS
    # =========================================================================
    
    @abstractmethod
    def get_all_vectors(self) -> NDArray:
        """
        Get all vectors as a matrix.
        
        Returns:
            Array of shape (n, dimension)
        """
        pass
    
    @abstractmethod
    def get_all_ids(self) -> List[str]:
        """Get all vector IDs."""
        pass
    
    @abstractmethod
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all metadata."""
        pass
    
    # =========================================================================
    # ITERATION
    # =========================================================================
    
    @abstractmethod
    def iter_ids(self) -> Iterator[str]:
        """Iterate over IDs."""
        pass
    
    @abstractmethod
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        """Iterate over all vectors."""
        pass
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    @abstractmethod
    def flush(self) -> None:
        """Flush pending writes to storage."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close storage and release resources."""
        pass
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    @abstractmethod
    def stats(self) -> StorageStats:
        """Get storage statistics."""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """
        Remove all vectors.
        
        Returns:
            Number of vectors removed
        """
        pass
    
    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================
    
    def __enter__(self) -> 'BaseStorage':
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
    
    def __len__(self) -> int:
        return self.size
    
    def __contains__(self, id: str) -> bool:
        return self.contains(id)