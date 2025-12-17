"""
Memory-mapped file storage for VectorDB.

Provides efficient storage for large vector datasets using
memory-mapped files, enabling datasets larger than RAM.
"""

import mmap
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import numpy as np

from .base import BaseStorage, StorageConfig, StorageStats, StorageMode


class MMapStorage(BaseStorage):
    """
    Memory-mapped file storage backend.
    
    Uses memory-mapped files for efficient access to large
    vector datasets without loading everything into RAM.
    
    Attributes:
        path: Storage directory path
        dimension: Vector dimension
        dtype: NumPy dtype for vectors
        
    Example:
        >>> storage = MMapStorage("./data", dimension=128)
        >>> storage.put("id1", vector, {"key": "value"})
        >>> vector, metadata = storage.get("id1")
    """
    
    # File format constants
    VECTORS_FILE = "vectors.mmap"
    INDEX_FILE = "index.bin"
    METADATA_FILE = "metadata.bin"
    HEADER_SIZE = 64  # bytes
    
    def __init__(
        self,
        path: str,
        dimension: int,
        dtype: np.dtype = np.float32,
        initial_capacity: int = 10000,
        growth_factor: float = 2.0,
        **kwargs
    ):
        """
        Initialize memory-mapped storage.
        
        Args:
            path: Directory path for storage files
            dimension: Vector dimension
            dtype: NumPy data type for vectors
            initial_capacity: Initial vector capacity
            growth_factor: Capacity growth factor when resizing
        """
        self._path = Path(path)
        self._dimension = dimension
        self._dtype = np.dtype(dtype)
        self._initial_capacity = initial_capacity
        self._growth_factor = growth_factor
        
        self._vector_size = dimension * self._dtype.itemsize
        self._capacity = 0
        self._count = 0
        
        self._mmap: Optional[mmap.mmap] = None
        self._file = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._metadata: Dict[str, dict] = {}
        self._free_indices: List[int] = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize or load existing storage."""
        self._path.mkdir(parents=True, exist_ok=True)
        vectors_path = self._path / self.VECTORS_FILE
        
        if vectors_path.exists():
            self._load()
        else:
            self._create(self._initial_capacity)
    
    def _create(self, capacity: int):
        """Create new memory-mapped storage."""
        vectors_path = self._path / self.VECTORS_FILE
        
        # Calculate file size
        file_size = self.HEADER_SIZE + capacity * self._vector_size
        
        # Create the file
        with open(vectors_path, "wb") as f:
            # Write header
            f.write(struct.pack("<I", self._dimension))
            f.write(struct.pack("<I", capacity))
            f.write(struct.pack("<I", 0))  # count
            f.write(b"\x00" * (self.HEADER_SIZE - 12))  # padding
            
            # Allocate space for vectors
            f.seek(file_size - 1)
            f.write(b"\x00")
        
        self._capacity = capacity
        self._count = 0
        self._open_mmap()
    
    def _load(self):
        """Load existing memory-mapped storage."""
        vectors_path = self._path / self.VECTORS_FILE
        
        # Read header
        with open(vectors_path, "rb") as f:
            dimension = struct.unpack("<I", f.read(4))[0]
            capacity = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<I", f.read(4))[0]
        
        if dimension != self._dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self._dimension}, "
                f"found {dimension}"
            )
        
        self._capacity = capacity
        self._count = count
        self._open_mmap()
        
        # Load index
        self._load_index()
        self._load_metadata()
    
    def _open_mmap(self):
        """Open or reopen the memory-mapped file."""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()
        
        vectors_path = self._path / self.VECTORS_FILE
        self._file = open(vectors_path, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), 0)
    
    def _load_index(self):
        """Load ID to index mapping."""
        index_path = self._path / self.INDEX_FILE
        if not index_path.exists():
            return
        
        with open(index_path, "rb") as f:
            count = struct.unpack("<I", f.read(4))[0]
            for _ in range(count):
                idx = struct.unpack("<I", f.read(4))[0]
                id_len = struct.unpack("<I", f.read(4))[0]
                id_str = f.read(id_len).decode("utf-8")
                self._id_to_idx[id_str] = idx
                self._idx_to_id[idx] = id_str
    
    def _save_index(self):
        """Save ID to index mapping."""
        index_path = self._path / self.INDEX_FILE
        
        with open(index_path, "wb") as f:
            f.write(struct.pack("<I", len(self._id_to_idx)))
            for id_str, idx in self._id_to_idx.items():
                id_bytes = id_str.encode("utf-8")
                f.write(struct.pack("<I", idx))
                f.write(struct.pack("<I", len(id_bytes)))
                f.write(id_bytes)
    
    def _load_metadata(self):
        """Load metadata from file."""
        import json
        metadata_path = self._path / self.METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)
    
    def _save_metadata(self):
        """Save metadata to file."""
        import json
        metadata_path = self._path / self.METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f)
    
    def _resize(self, new_capacity: int):
        """Resize the storage to new capacity."""
        vectors_path = self._path / self.VECTORS_FILE
        
        # Close current mmap
        self._mmap.close()
        self._file.close()
        
        # Calculate new file size
        new_size = self.HEADER_SIZE + new_capacity * self._vector_size
        
        # Resize file
        with open(vectors_path, "r+b") as f:
            f.seek(0, 2)  # End of file
            current_size = f.tell()
            if new_size > current_size:
                f.seek(new_size - 1)
                f.write(b"\x00")
            
            # Update header
            f.seek(4)
            f.write(struct.pack("<I", new_capacity))
        
        self._capacity = new_capacity
        self._open_mmap()
    
    def _get_offset(self, idx: int) -> int:
        """Get byte offset for vector at index."""
        return self.HEADER_SIZE + idx * self._vector_size
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    def put(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a vector with optional metadata.
        
        Args:
            id: Unique identifier
            vector: Vector to store
            metadata: Optional metadata dictionary
        """
        if id in self._id_to_idx:
            # Update existing
            idx = self._id_to_idx[id]
        else:
            # Get new index
            if self._free_indices:
                idx = self._free_indices.pop()
            else:
                # Check capacity
                if self._count >= self._capacity:
                    new_capacity = int(self._capacity * self._growth_factor)
                    self._resize(new_capacity)
                idx = self._count
                self._count += 1
            
            self._id_to_idx[id] = idx
            self._idx_to_id[idx] = id
        
        # Write vector to mmap
        offset = self._get_offset(idx)
        vec_bytes = vector.astype(self._dtype).tobytes()
        self._mmap[offset:offset + self._vector_size] = vec_bytes
        
        # Store metadata
        if metadata:
            self._metadata[id] = metadata
    
    def get(self, id: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Retrieve a vector and its metadata.
        
        Args:
            id: Vector identifier
            
        Returns:
            Tuple of (vector, metadata) or (None, None) if not found
        """
        if id not in self._id_to_idx:
            return None, None
        
        idx = self._id_to_idx[id]
        offset = self._get_offset(idx)
        
        vec_bytes = self._mmap[offset:offset + self._vector_size]
        vector = np.frombuffer(vec_bytes, dtype=self._dtype).copy()
        metadata = self._metadata.get(id)
        
        return vector, metadata
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector.
        
        Args:
            id: Vector identifier
            
        Returns:
            True if deleted, False if not found
        """
        if id not in self._id_to_idx:
            return False
        
        idx = self._id_to_idx.pop(id)
        del self._idx_to_id[idx]
        self._free_indices.append(idx)
        self._metadata.pop(id, None)
        
        return True
    
    def exists(self, id: str) -> bool:
        """Check if vector exists."""
        return id in self._id_to_idx
    
    def count(self) -> int:
        """Get number of stored vectors."""
        return len(self._id_to_idx)
    
    def get_all_vectors(self) -> np.ndarray:
        """Get all vectors as a matrix."""
        n = len(self._id_to_idx)
        if n == 0:
            return np.empty((0, self._dimension), dtype=self._dtype)
        
        vectors = np.empty((n, self._dimension), dtype=self._dtype)
        for i, id_str in enumerate(self._id_to_idx.keys()):
            vec, _ = self.get(id_str)
            vectors[i] = vec
        
        return vectors
    
    def get_all_ids(self) -> List[str]:
        """Get all vector IDs."""
        return list(self._id_to_idx.keys())
    
    def flush(self):
        """Flush changes to disk."""
        if self._mmap:
            self._mmap.flush()
        self._save_index()
        self._save_metadata()
        
        # Update header count
        if self._file:
            self._file.seek(8)
            self._file.write(struct.pack("<I", self._count))
            self._file.flush()
    
    def close(self):
        """Close the storage."""
        self.flush()
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
    
    def stats(self) -> StorageStats:
        """Get storage statistics."""
        vectors_path = self._path / self.VECTORS_FILE
        file_size = vectors_path.stat().st_size if vectors_path.exists() else 0
        
        return StorageStats(
            count=len(self._id_to_idx),
            dimension=self._dimension,
            storage_bytes=file_size,
            metadata_count=len(self._metadata),
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __len__(self) -> int:
        return self.count()
    
    def __contains__(self, id: str) -> bool:
        return self.exists(id)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._id_to_idx.keys())
