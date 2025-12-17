"""
Vector record and batch definitions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Iterator
import uuid
import time


@dataclass
class VectorRecord:
    """
    A single vector with its ID and metadata.
    
    Attributes:
        id: Unique identifier for the vector
        vector: The embedding vector (numpy array)
        metadata: Optional dictionary of metadata
        timestamp: Creation/update timestamp
    
    Example:
        >>> record = VectorRecord(
        ...     id="doc_001",
        ...     vector=np.array([0.1, 0.2, 0.3]),
        ...     metadata={"category": "science", "year": 2023}
        ... )
    """
    
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate and convert vector to proper format."""
        # Generate ID if not provided
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Convert vector to numpy array if needed
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)
        
        # Ensure float32 for memory efficiency
        if self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)
        
        # Ensure 1D vector
        if self.vector.ndim != 1:
            raise ValueError(
                f"Vector must be 1-dimensional, got {self.vector.ndim} dimensions"
            )
        
        # Validate vector is not empty
        if len(self.vector) == 0:
            raise ValueError("Vector cannot be empty")
        
        # Check for NaN or Inf values
        if not np.isfinite(self.vector).all():
            raise ValueError("Vector contains NaN or Inf values")
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the vector."""
        return len(self.vector)
    
    @property
    def norm(self) -> float:
        """Return the L2 norm of the vector."""
        return float(np.linalg.norm(self.vector))
    
    def normalize(self) -> VectorRecord:
        """
        Return a new record with normalized vector (unit length).
        
        Returns:
            New VectorRecord with normalized vector
        """
        norm = self.norm
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        
        return VectorRecord(
            id=self.id,
            vector=self.vector / norm,
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
        )
    
    def copy(self) -> VectorRecord:
        """Create a deep copy of this record."""
        return VectorRecord(
            id=self.id,
            vector=self.vector.copy(),
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
        )
    
    def update_metadata(self, new_metadata: Dict[str, Any]) -> VectorRecord:
        """
        Return a new record with updated metadata.
        
        Args:
            new_metadata: Metadata fields to add/update
            
        Returns:
            New VectorRecord with merged metadata
        """
        merged = {**self.metadata, **new_metadata}
        return VectorRecord(
            id=self.id,
            vector=self.vector.copy(),
            metadata=merged,
            timestamp=time.time(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary (for serialization)."""
        return {
            "id": self.id,
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VectorRecord:
        """Create record from dictionary."""
        return cls(
            id=data["id"],
            vector=np.array(data["vector"], dtype=np.float32),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )
    
    def __repr__(self) -> str:
        return (
            f"VectorRecord(id='{self.id}', "
            f"dim={self.dimension}, "
            f"norm={self.norm:.4f}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorRecord):
            return False
        return (
            self.id == other.id and
            np.array_equal(self.vector, other.vector) and
            self.metadata == other.metadata
        )
    
    def __hash__(self) -> int:
        return hash(self.id)


class VectorBatch:
    """
    A batch of vectors for efficient bulk operations.
    
    Stores vectors in a contiguous numpy array for vectorized operations.
    
    Example:
        >>> batch = VectorBatch(dimension=128)
        >>> batch.add("id1", np.random.randn(128), {"category": "A"})
        >>> batch.add("id2", np.random.randn(128), {"category": "B"})
        >>> vectors = batch.vectors  # (2, 128) numpy array
    """
    
    def __init__(self, dimension: int, initial_capacity: int = 1000):
        """
        Initialize a vector batch.
        
        Args:
            dimension: Dimension of vectors
            initial_capacity: Initial capacity for pre-allocation
        """
        self.dimension = dimension
        self._capacity = initial_capacity
        self._size = 0
        
        # Pre-allocate arrays
        self._vectors = np.zeros((initial_capacity, dimension), dtype=np.float32)
        self._ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._timestamps: List[float] = []
    
    def add(
        self, 
        id: str, 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a vector to the batch.
        
        Args:
            id: Unique identifier
            vector: The embedding vector
            metadata: Optional metadata dictionary
        """
        # Convert and validate
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} doesn't match batch dimension {self.dimension}"
            )
        
        # Expand capacity if needed
        if self._size >= self._capacity:
            self._expand()
        
        # Add to batch
        self._vectors[self._size] = vector
        self._ids.append(id)
        self._metadata.append(metadata or {})
        self._timestamps.append(time.time())
        self._size += 1
    
    def add_record(self, record: VectorRecord) -> None:
        """Add a VectorRecord to the batch."""
        self.add(record.id, record.vector, record.metadata)
    
    def _expand(self, factor: float = 2.0) -> None:
        """Expand the internal capacity."""
        new_capacity = int(self._capacity * factor)
        new_vectors = np.zeros((new_capacity, self.dimension), dtype=np.float32)
        new_vectors[:self._size] = self._vectors[:self._size]
        self._vectors = new_vectors
        self._capacity = new_capacity
    
    @property
    def vectors(self) -> np.ndarray:
        """Return all vectors as a 2D numpy array."""
        return self._vectors[:self._size]
    
    @property
    def ids(self) -> List[str]:
        """Return all IDs."""
        return self._ids.copy()
    
    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Return all metadata."""
        return self._metadata.copy()
    
    def __len__(self) -> int:
        return self._size
    
    def __iter__(self) -> Iterator[VectorRecord]:
        """Iterate over records in the batch."""
        for i in range(self._size):
            yield VectorRecord(
                id=self._ids[i],
                vector=self._vectors[i].copy(),
                metadata=self._metadata[i],
                timestamp=self._timestamps[i],
            )
    
    def __getitem__(self, index: int) -> VectorRecord:
        """Get record by index."""
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size})")
        
        return VectorRecord(
            id=self._ids[index],
            vector=self._vectors[index].copy(),
            metadata=self._metadata[index],
            timestamp=self._timestamps[index],
        )
    
    def get_by_id(self, id: str) -> Optional[VectorRecord]:
        """Get record by ID (linear search)."""
        try:
            idx = self._ids.index(id)
            return self[idx]
        except ValueError:
            return None
    
    def to_records(self) -> List[VectorRecord]:
        """Convert batch to list of VectorRecords."""
        return list(self)
    
    @classmethod
    def from_records(cls, records: List[VectorRecord]) -> VectorBatch:
        """Create batch from list of VectorRecords."""
        if not records:
            raise ValueError("Cannot create batch from empty list")
        
        dimension = records[0].dimension
        batch = cls(dimension=dimension, initial_capacity=len(records))
        
        for record in records:
            batch.add_record(record)
        
        return batch
    
    @classmethod
    def from_numpy(
        cls,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorBatch:
        """
        Create batch from numpy array.
        
        Args:
            vectors: 2D numpy array of shape (n, dimension)
            ids: Optional list of IDs (auto-generated if not provided)
            metadata: Optional list of metadata dicts
        """
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")
        
        n, dimension = vectors.shape
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        elif len(ids) != n:
            raise ValueError(f"Number of IDs ({len(ids)}) doesn't match vectors ({n})")
        
        # Default metadata
        if metadata is None:
            metadata = [{} for _ in range(n)]
        elif len(metadata) != n:
            raise ValueError(f"Number of metadata ({len(metadata)}) doesn't match vectors ({n})")
        
        batch = cls(dimension=dimension, initial_capacity=n)
        
        for i in range(n):
            batch.add(ids[i], vectors[i], metadata[i])
        
        return batch
    
    def normalize_all(self) -> VectorBatch:
        """Return new batch with all vectors normalized."""
        norms = np.linalg.norm(self._vectors[:self._size], axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        
        normalized = self._vectors[:self._size] / norms
        
        return VectorBatch.from_numpy(
            normalized,
            ids=self._ids.copy(),
            metadata=[m.copy() for m in self._metadata],
        )
    
    def __repr__(self) -> str:
        return f"VectorBatch(size={self._size}, dimension={self.dimension})"


# Type alias for convenience
Vector = Union[np.ndarray, List[float]]


def validate_vector(
    vector: Vector,
    expected_dimension: Optional[int] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Validate and convert a vector to proper format.
    
    Args:
        vector: Input vector (list or numpy array)
        expected_dimension: Expected dimension (optional)
        normalize: Whether to normalize the vector
        
    Returns:
        Validated numpy array (float32)
        
    Raises:
        ValueError: If vector is invalid
    """
    # Convert to numpy
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector, dtype=np.float32)
    
    # Ensure float32
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    
    # Validate dimension
    if vector.ndim != 1:
        raise ValueError(f"Vector must be 1D, got {vector.ndim}D")
    
    if len(vector) == 0:
        raise ValueError("Vector cannot be empty")
    
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError(
            f"Vector dimension {len(vector)} doesn't match expected {expected_dimension}"
        )
    
    # Check for invalid values
    if not np.isfinite(vector).all():
        raise ValueError("Vector contains NaN or Inf values")
    
    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        vector = vector / norm
    
    return vector