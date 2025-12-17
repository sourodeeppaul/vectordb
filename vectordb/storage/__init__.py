"""
Storage backends for VectorDB.

Available Storage Backends:
    - MemoryStorage: In-memory storage (fast, volatile)
    - DiskStorage: Persistent disk storage with optional mmap
    - MMapStorage: Memory-mapped file storage (large datasets)

Example:
    >>> from vectordb.storage import DiskStorage
    >>> 
    >>> # Create disk storage
    >>> storage = DiskStorage("./data", dimension=128)
    >>> 
    >>> # Store vectors
    >>> storage.put("id1", vector1, {"key": "value"})
    >>> 
    >>> # Retrieve vectors
    >>> vector, metadata = storage.get("id1")
    >>> 
    >>> # Flush to disk
    >>> storage.flush()
"""

from .base import (
    BaseStorage,
    StorageConfig,
    StorageStats,
    StorageMode,
)

from .memory import MemoryStorage
from .disk import DiskStorage
from .serialization import (
    VectorSerializer,
    serialize_vector,
    deserialize_vector,
    serialize_metadata,
    deserialize_metadata,
)
from .format import (
    FileHeader,
    FileFormat,
    MAGIC_NUMBER,
    VERSION,
)

__all__ = [
    # Base
    "BaseStorage",
    "StorageConfig",
    "StorageStats",
    "StorageMode",
    # Implementations
    "MemoryStorage",
    "DiskStorage",
    # Serialization
    "VectorSerializer",
    "serialize_vector",
    "deserialize_vector",
    "serialize_metadata",
    "deserialize_metadata",
    # Format
    "FileHeader",
    "FileFormat",
    "MAGIC_NUMBER",
    "VERSION",
    # Factory
    "create_storage",
]


def create_storage(
    storage_type: str,
    path: str = None,
    dimension: int = None,
    **kwargs
) -> BaseStorage:
    """
    Factory function to create storage backend.
    
    Args:
        storage_type: "memory" or "disk"
        path: Path for disk storage
        dimension: Vector dimension
        **kwargs: Storage-specific options
        
    Returns:
        Storage instance
    """
    storage_type = storage_type.lower()
    
    if storage_type == "memory":
        return MemoryStorage(dimension=dimension, **kwargs)
    elif storage_type == "disk":
        if path is None:
            raise ValueError("path required for disk storage")
        return DiskStorage(path=path, dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")