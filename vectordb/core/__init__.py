"""
Core components for VectorDB.
"""

from .vector import VectorRecord, VectorBatch, validate_vector
from .collection import (
    Collection,
    CollectionConfig,
    CollectionStats,
    SearchResult,
    MetadataIndex,
    IndexType,
)
from .database import VectorDB
from .exceptions import (
    VectorDBError,
    CollectionError,
    CollectionNotFoundError,
    CollectionExistsError,
    VectorError,
    VectorNotFoundError,
    VectorExistsError,
    DimensionMismatchError,
    ValidationError,
    IndexError,
    IndexNotTrainedError,
    StorageError,
    SerializationError,
)

__all__ = [
    # Vector
    "VectorRecord",
    "VectorBatch",
    "validate_vector",
    # Collection
    "Collection",
    "CollectionConfig",
    "CollectionStats",
    "SearchResult",
    "MetadataIndex",
    "IndexType",
    # Database
    "VectorDB",
    # Exceptions
    "VectorDBError",
    "CollectionError",
    "CollectionNotFoundError",
    "CollectionExistsError",
    "VectorError",
    "VectorNotFoundError",
    "VectorExistsError",
    "DimensionMismatchError",
    "ValidationError",
    "IndexError",
    "IndexNotTrainedError",
    "StorageError",
    "SerializationError",
]