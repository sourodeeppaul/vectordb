"""
In-memory storage backend.

Fast volatile storage for vectors and metadata.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Any, Tuple, Iterator
import threading

from .base import BaseStorage, StorageConfig, StorageStats


class MemoryStorage(BaseStorage):
    """
    In-memory vector storage.
    
    Fast but volatile - data is lost when the process exits.
    Use for:
    - Development and testing
    - Temporary data
    - Small datasets that fit in memory
    
    Example:
        >>> storage = MemoryStorage(dimension=128)
        >>> storage.put("id1", vector, {"key": "value"})
        >>> vector, metadata = storage.get("id1")
    """
    
    def __init__(
        self,
        dimension: int,
        initial_capacity: int = 1000,
        growth_factor: float = 2.0,
        **kwargs,
    ):
        """
        Initialize memory storage.
        
        Args:
            dimension: Vector dimension
            initial_capacity: Initial capacity for pre-allocation
            growth_factor: Growth factor when expanding
        """
        super().__init__(dimension, **kwargs)
        
        self._initial_capacity = initial_capacity
        self._growth_factor = growth_factor
        
        # Storage
        self._vectors = np.zeros(
            (initial_capacity, dimension), dtype=np.float32
        )
        self._capacity = initial_capacity
        self._size = 0
        
        # ID mappings
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        
        # Metadata
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Free list for deleted indices
        self._free_indices: List[int] = []
        
        # Statistics
        self._reads = 0
        self._writes = 0
    
    @property
    def size(self) -> int:
        return self._size
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    def put(
        self,
        id: str,
        vector: NDArray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a vector."""
        vector = self._validate_vector(vector)
        
        with self._lock:
            if id in self._id_to_index:
                # Update existing
                self.update(id, vector, metadata)
                return
            
            # Get index
            if self._free_indices:
                index = self._free_indices.pop()
            else:
                index = self._size
                if index >= self._capacity:
                    self._expand()
            
            # Store
            self._vectors[index] = vector
            self._id_to_index[id] = index
            self._index_to_id[index] = id
            self._metadata[id] = metadata or {}
            
            self._size += 1
            self._writes += 1
    
    def get(self, id: str) -> Optional[Tuple[NDArray, Dict[str, Any]]]:
        """Retrieve a vector."""
        with self._lock:
            if id not in self._id_to_index:
                return None
            
            index = self._id_to_index[id]
            vector = self._vectors[index].copy()
            metadata = self._metadata.get(id, {}).copy()
            
            self._reads += 1
            return vector, metadata
    
    def delete(self, id: str) -> bool:
        """Delete a vector."""
        with self._lock:
            if id not in self._id_to_index:
                return False
            
            index = self._id_to_index[id]
            
            # Remove from mappings
            del self._id_to_index[id]
            del self._index_to_id[index]
            
            if id in self._metadata:
                del self._metadata[id]
            
            # Add to free list
            self._free_indices.append(index)
            self._size -= 1
            
            return True
    
    def contains(self, id: str) -> bool:
        """Check if ID exists."""
        return id in self._id_to_index
    
    def update(
        self,
        id: str,
        vector: Optional[NDArray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector."""
        with self._lock:
            if id not in self._id_to_index:
                return False
            
            index = self._id_to_index[id]
            
            if vector is not None:
                vector = self._validate_vector(vector)
                self._vectors[index] = vector
            
            if metadata is not None:
                self._metadata[id] = metadata
            
            self._writes += 1
            return True
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    def put_batch(
        self,
        ids: List[str],
        vectors: NDArray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Store multiple vectors."""
        vectors = self._validate_vectors(vectors)
        n = len(ids)
        
        if len(vectors) != n:
            raise ValueError(f"IDs count ({n}) != vectors count ({len(vectors)})")
        
        with self._lock:
            # Ensure capacity
            needed = n - len(self._free_indices)
            while self._size + needed > self._capacity:
                self._expand()
            
            for i in range(n):
                self.put(
                    ids[i],
                    vectors[i],
                    metadata[i] if metadata else None,
                )
        
        return n
    
    def get_batch(
        self,
        ids: List[str]
    ) -> List[Optional[Tuple[NDArray, Dict[str, Any]]]]:
        """Retrieve multiple vectors."""
        return [self.get(id) for id in ids]
    
    def delete_batch(self, ids: List[str]) -> int:
        """Delete multiple vectors."""
        count = 0
        for id in ids:
            if self.delete(id):
                count += 1
        return count
    
    # =========================================================================
    # BULK ACCESS
    # =========================================================================
    
    def get_all_vectors(self) -> NDArray:
        """Get all vectors as a matrix."""
        with self._lock:
            if self._size == 0:
                return np.zeros((0, self._dimension), dtype=np.float32)
            
            # Get vectors in ID order
            result = np.zeros((self._size, self._dimension), dtype=np.float32)
            for i, id in enumerate(self._id_to_index.keys()):
                index = self._id_to_index[id]
                result[i] = self._vectors[index]
            
            return result
    
    def get_all_ids(self) -> List[str]:
        """Get all vector IDs."""
        with self._lock:
            return list(self._id_to_index.keys())
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all metadata."""
        with self._lock:
            return {id: meta.copy() for id, meta in self._metadata.items()}
    
    # Get vectors by indices (for efficient index access)
    def get_vectors_by_indices(self, indices: List[int]) -> NDArray:
        """Get vectors by internal indices."""
        with self._lock:
            return self._vectors[indices].copy()
    
    def get_contiguous_vectors(self) -> Tuple[NDArray, List[str]]:
        """
        Get all vectors as a contiguous array with their IDs.
        
        More efficient for index operations.
        
        Returns:
            Tuple of (vectors array, list of IDs in same order)
        """
        with self._lock:
            if self._size == 0:
                return np.zeros((0, self._dimension), dtype=np.float32), []
            
            ids = []
            indices = []
            
            for id, index in self._id_to_index.items():
                ids.append(id)
                indices.append(index)
            
            vectors = self._vectors[indices].copy()
            return vectors, ids
    
    # =========================================================================
    # ITERATION
    # =========================================================================
    
    def iter_ids(self) -> Iterator[str]:
        """Iterate over IDs."""
        with self._lock:
            yield from list(self._id_to_index.keys())
    
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        """Iterate over all vectors."""
        with self._lock:
            for id in list(self._id_to_index.keys()):
                index = self._id_to_index[id]
                vector = self._vectors[index].copy()
                metadata = self._metadata.get(id, {}).copy()
                yield id, vector, metadata
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def flush(self) -> None:
        """No-op for memory storage."""
        pass
    
    def close(self) -> None:
        """No-op for memory storage."""
        pass
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def stats(self) -> StorageStats:
        """Get storage statistics."""
        with self._lock:
            memory = self._vectors.nbytes
            memory += sum(len(id) * 2 for id in self._id_to_index)
            memory += sum(len(str(m)) for m in self._metadata.values())
            
            return StorageStats(
                vector_count=self._size,
                dimension=self._dimension,
                memory_bytes=memory,
                disk_bytes=0,
                reads=self._reads,
                writes=self._writes,
            )
    
    def clear(self) -> int:
        """Remove all vectors."""
        with self._lock:
            count = self._size
            
            self._vectors = np.zeros(
                (self._initial_capacity, self._dimension), dtype=np.float32
            )
            self._capacity = self._initial_capacity
            self._size = 0
            
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._metadata.clear()
            self._free_indices.clear()
            
            return count
    
    # =========================================================================
    # INTERNAL
    # =========================================================================
    
    def _validate_vector(self, vector: NDArray) -> NDArray:
        """Validate and normalize vector."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} != {self._dimension}"
            )
        
        return vector
    
    def _validate_vectors(self, vectors: NDArray) -> NDArray:
        """Validate batch of vectors."""
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vectors shape {vectors.shape} invalid for dimension {self._dimension}"
            )
        
        return vectors
    
    def _expand(self) -> None:
        """Expand storage capacity."""
        new_capacity = int(self._capacity * self._growth_factor)
        new_vectors = np.zeros(
            (new_capacity, self._dimension), dtype=np.float32
        )
        new_vectors[:self._capacity] = self._vectors
        self._vectors = new_vectors
        self._capacity = new_capacity
    
    def __repr__(self) -> str:
        return f"MemoryStorage(dimension={self._dimension}, size={self._size})"