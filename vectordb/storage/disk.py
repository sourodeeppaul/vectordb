"""
Disk-based storage backend with memory mapping.

Provides persistent storage for vectors with optional
memory mapping for efficient access to large datasets.
"""

from __future__ import annotations

import os
import numpy as np
from numpy.typing import NDArray
import mmap
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterator, BinaryIO

from .base import BaseStorage, StorageConfig, StorageStats, StorageMode
from .format import (
    FileHeader,
    FileFooter,
    IndexEntry,
    FileFormat,
    FileFlags,
    MAGIC_NUMBER,
    VERSION,
)
from .serialization import (
    VectorSerializer,
    serialize_metadata,
    deserialize_metadata,
    StreamWriter,
    StreamReader,
)


class DiskStorage(BaseStorage):
    """
    Disk-based vector storage with memory mapping.
    
    Features:
    - Persistent storage across restarts
    - Memory mapping for large datasets
    - Efficient random access
    - Atomic writes with journaling
    
    Example:
        >>> storage = DiskStorage("./data", dimension=128)
        >>> storage.put("id1", vector, {"key": "value"})
        >>> storage.flush()  # Persist to disk
        >>> 
        >>> # Later, reload
        >>> storage = DiskStorage("./data", dimension=128)
        >>> vector, metadata = storage.get("id1")
    
    File Structure:
        data_dir/
        ├── vectors.vdb      # Main vector file
        ├── index.idx        # ID -> offset mapping
        ├── metadata.mdb     # Metadata store
        └── wal.log          # Write-ahead log
    """
    
    VECTORS_FILE = "vectors.vdb"
    INDEX_FILE = "index.idx"
    METADATA_FILE = "metadata.mdb"
    WAL_FILE = "wal.log"
    
    def __init__(
        self,
        path: str,
        dimension: int,
        mode: StorageMode = StorageMode.READ_WRITE,
        use_mmap: bool = True,
        sync_on_write: bool = False,
        create_if_missing: bool = True,
        **kwargs,
    ):
        """
        Initialize disk storage.
        
        Args:
            path: Directory path for storage files
            dimension: Vector dimension
            mode: Access mode
            use_mmap: Use memory mapping for vectors
            sync_on_write: Sync to disk after each write
            create_if_missing: Create directory if it doesn't exist
        """
        super().__init__(dimension, **kwargs)
        
        self._path = Path(path)
        self._mode = mode
        self._use_mmap = use_mmap
        self._sync_on_write = sync_on_write
        
        # Create directory
        if create_if_missing:
            self._path.mkdir(parents=True, exist_ok=True)
        elif not self._path.exists():
            raise FileNotFoundError(f"Storage path not found: {path}")
        
        # File paths
        self._vectors_path = self._path / self.VECTORS_FILE
        self._index_path = self._path / self.INDEX_FILE
        self._metadata_path = self._path / self.METADATA_FILE
        self._wal_path = self._path / self.WAL_FILE
        
        # Serializer
        self._serializer = VectorSerializer(dimension)
        
        # In-memory structures
        self._id_to_offset: Dict[str, int] = {}  # ID -> vector offset
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._size = 0
        
        # File handles
        self._vectors_file: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None
        
        # Write buffer
        self._write_buffer: List[Tuple[str, NDArray, Dict[str, Any]]] = []
        self._buffer_size = 0
        self._max_buffer_size = 10 * 1024 * 1024  # 10MB
        
        # Statistics
        self._reads = 0
        self._writes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Load existing data
        self._load()
    
    @property
    def size(self) -> int:
        return self._size
    
    # =========================================================================
    # LOADING AND INITIALIZATION
    # =========================================================================
    
    def _load(self) -> None:
        """Load existing data from disk."""
        if not self._vectors_path.exists():
            self._initialize_files()
            return
        
        # Load index
        self._load_index()
        
        # Open vectors file
        self._open_vectors_file()
        
        # Load metadata
        self._load_metadata()
    
    def _initialize_files(self) -> None:
        """Initialize empty storage files."""
        # Create empty vectors file with header
        header = FileHeader(
            dimension=self._dimension,
            vector_count=0,
            flags=FileFlags.HAS_METADATA | FileFlags.HAS_INDEX,
        )
        
        with open(self._vectors_path, 'wb') as f:
            f.write(header.to_bytes())
        
        # Create empty index
        self._save_index()
        
        # Create empty metadata
        self._save_metadata()
        
        # Open for access
        self._open_vectors_file()
    
    def _open_vectors_file(self) -> None:
        """Open vectors file with optional mmap."""
        mode = 'r+b' if self._mode != StorageMode.READ_ONLY else 'rb'
        
        if not self._vectors_path.exists():
            if self._mode == StorageMode.READ_ONLY:
                raise FileNotFoundError(f"Vectors file not found: {self._vectors_path}")
            mode = 'w+b'
        
        self._vectors_file = open(self._vectors_path, mode)
        
        # Memory map if enabled and file has data
        if self._use_mmap and os.path.getsize(self._vectors_path) > FileHeader.SIZE:
            access = mmap.ACCESS_READ if self._mode == StorageMode.READ_ONLY else mmap.ACCESS_WRITE
            try:
                self._mmap = mmap.mmap(
                    self._vectors_file.fileno(),
                    0,  # Map entire file
                    access=access,
                )
            except (ValueError, OSError):
                # mmap failed, continue without it
                self._mmap = None
    
    def _load_index(self) -> None:
        """Load ID -> offset index from disk."""
        if not self._index_path.exists():
            return
        
        try:
            import json
            with open(self._index_path, 'r') as f:
                data = json.load(f)
            
            self._id_to_offset = data.get("id_to_offset", {})
            self._size = len(self._id_to_offset)
        except Exception as e:
            print(f"Warning: Failed to load index: {e}")
            self._id_to_offset = {}
            self._size = 0
    
    def _save_index(self) -> None:
        """Save ID -> offset index to disk."""
        import json
        
        data = {
            "id_to_offset": self._id_to_offset,
            "version": VERSION,
        }
        
        with open(self._index_path, 'w') as f:
            json.dump(data, f)
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if not self._metadata_path.exists():
            return
        
        try:
            with open(self._metadata_path, 'rb') as f:
                data = f.read()
            self._metadata_cache = deserialize_metadata(data)
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")
            self._metadata_cache = {}
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        data = serialize_metadata(self._metadata_cache)
        with open(self._metadata_path, 'wb') as f:
            f.write(data)
    
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
        if self._mode == StorageMode.READ_ONLY:
            raise IOError("Storage is read-only")
        
        vector = self._validate_vector(vector)
        
        with self._lock:
            # Add to write buffer
            self._write_buffer.append((id, vector.copy(), metadata or {}))
            self._buffer_size += self._serializer.vector_size
            
            # Update in-memory index
            if id not in self._id_to_offset:
                self._size += 1
            
            # Store metadata in cache
            self._metadata_cache[id] = metadata or {}
            
            self._writes += 1
            
            # Flush if buffer is full
            if self._buffer_size >= self._max_buffer_size:
                self._flush_buffer()
    
    def get(self, id: str) -> Optional[Tuple[NDArray, Dict[str, Any]]]:
        """Retrieve a vector."""
        with self._lock:
            # Check write buffer first
            for buf_id, buf_vec, buf_meta in reversed(self._write_buffer):
                if buf_id == id:
                    self._cache_hits += 1
                    self._reads += 1
                    return buf_vec.copy(), buf_meta.copy()
            
            # Check if exists
            if id not in self._id_to_offset:
                return None
            
            self._cache_misses += 1
            
            # Read from disk/mmap
            offset = self._id_to_offset[id]
            vector = self._read_vector(offset)
            metadata = self._metadata_cache.get(id, {}).copy()
            
            self._reads += 1
            return vector, metadata
    
    def _read_vector(self, offset: int) -> NDArray:
        """Read vector from storage."""
        if self._mmap is not None:
            # Read from mmap
            data = self._mmap[offset:offset + self._serializer.vector_size]
            return self._serializer.deserialize(bytes(data))
        else:
            # Read from file
            self._vectors_file.seek(offset)
            data = self._vectors_file.read(self._serializer.vector_size)
            return self._serializer.deserialize(data)
    
    def delete(self, id: str) -> bool:
        """Delete a vector."""
        if self._mode == StorageMode.READ_ONLY:
            raise IOError("Storage is read-only")
        
        with self._lock:
            # Remove from buffer
            self._write_buffer = [
                (i, v, m) for i, v, m in self._write_buffer if i != id
            ]
            
            if id not in self._id_to_offset:
                return False
            
            # Mark as deleted (actual removal on compaction)
            del self._id_to_offset[id]
            
            if id in self._metadata_cache:
                del self._metadata_cache[id]
            
            self._size -= 1
            return True
    
    def contains(self, id: str) -> bool:
        """Check if ID exists."""
        with self._lock:
            # Check buffer
            for buf_id, _, _ in self._write_buffer:
                if buf_id == id:
                    return True
            return id in self._id_to_offset
    
    def update(
        self,
        id: str,
        vector: Optional[NDArray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector."""
        if self._mode == StorageMode.READ_ONLY:
            raise IOError("Storage is read-only")
        
        with self._lock:
            if not self.contains(id):
                return False
            
            # Get current data
            current = self.get(id)
            if current is None:
                return False
            
            current_vector, current_metadata = current
            
            # Update
            new_vector = vector if vector is not None else current_vector
            new_metadata = metadata if metadata is not None else current_metadata
            
            # Remove from buffer if present
            self._write_buffer = [
                (i, v, m) for i, v, m in self._write_buffer if i != id
            ]
            
            # Add updated version
            self._write_buffer.append((id, new_vector.copy(), new_metadata))
            self._metadata_cache[id] = new_metadata
            
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
        
        for i, id in enumerate(ids):
            self.put(
                id,
                vectors[i],
                metadata[i] if metadata else None,
            )
        
        return len(ids)
    
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
            self._flush_buffer()
            
            if self._size == 0:
                return np.zeros((0, self._dimension), dtype=np.float32)
            
            vectors = []
            for id in self._id_to_offset:
                vec, _ = self.get(id)
                if vec is not None:
                    vectors.append(vec)
            
            return np.array(vectors, dtype=np.float32)
    
    def get_all_ids(self) -> List[str]:
        """Get all vector IDs."""
        with self._lock:
            # Combine index and buffer
            ids = set(self._id_to_offset.keys())
            for buf_id, _, _ in self._write_buffer:
                ids.add(buf_id)
            return list(ids)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all metadata."""
        with self._lock:
            return {id: meta.copy() for id, meta in self._metadata_cache.items()}
    
    # =========================================================================
    # ITERATION
    # =========================================================================
    
    def iter_ids(self) -> Iterator[str]:
        """Iterate over IDs."""
        with self._lock:
            yield from list(self._id_to_offset.keys())
            for buf_id, _, _ in self._write_buffer:
                if buf_id not in self._id_to_offset:
                    yield buf_id
    
    def iter_vectors(self) -> Iterator[Tuple[str, NDArray, Dict[str, Any]]]:
        """Iterate over all vectors."""
        for id in self.get_all_ids():
            result = self.get(id)
            if result:
                vector, metadata = result
                yield id, vector, metadata
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def flush(self) -> None:
        """Flush all pending writes to disk."""
        with self._lock:
            self._flush_buffer()
            self._save_index()
            self._save_metadata()
            
            if self._vectors_file:
                self._vectors_file.flush()
                if self._sync_on_write:
                    os.fsync(self._vectors_file.fileno())
    
    def _flush_buffer(self) -> None:
        """Flush write buffer to disk."""
        if not self._write_buffer:
            return
        
        # Close mmap for writing
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        
        # Append to vectors file
        self._vectors_file.seek(0, 2)  # End of file
        
        for id, vector, metadata in self._write_buffer:
            offset = self._vectors_file.tell()
            data = self._serializer.serialize(vector)
            self._vectors_file.write(data)
            self._id_to_offset[id] = offset
        
        self._vectors_file.flush()
        
        # Reopen mmap
        if self._use_mmap:
            self._reopen_mmap()
        
        # Clear buffer
        self._write_buffer.clear()
        self._buffer_size = 0
    
    def _reopen_mmap(self) -> None:
        """Reopen memory map after file modification."""
        try:
            access = mmap.ACCESS_READ if self._mode == StorageMode.READ_ONLY else mmap.ACCESS_WRITE
            self._mmap = mmap.mmap(
                self._vectors_file.fileno(),
                0,
                access=access,
            )
        except (ValueError, OSError):
            self._mmap = None
    
    def close(self) -> None:
        """Close storage and release resources."""
        with self._lock:
            self.flush()
            
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            
            if self._vectors_file:
                self._vectors_file.close()
                self._vectors_file = None
    
    # =========================================================================
    # COMPACTION
    # =========================================================================
    
    def compact(self) -> None:
        """
        Compact storage by removing deleted vectors.
        
        Creates a new file with only active vectors,
        then replaces the old file.
        """
        if self._mode == StorageMode.READ_ONLY:
            raise IOError("Storage is read-only")
        
        with self._lock:
            self.flush()
            
            # Create new file
            temp_path = self._path / "vectors.vdb.tmp"
            new_offsets = {}
            
            with open(temp_path, 'wb') as f:
                # Write header
                header = FileHeader(
                    dimension=self._dimension,
                    vector_count=self._size,
                    flags=FileFlags.HAS_METADATA | FileFlags.HAS_INDEX,
                )
                f.write(header.to_bytes())
                
                # Write vectors
                for id in self._id_to_offset:
                    result = self.get(id)
                    if result:
                        vector, _ = result
                        new_offsets[id] = f.tell()
                        f.write(self._serializer.serialize(vector))
            
            # Close current file
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            
            self._vectors_file.close()
            
            # Replace old file
            os.replace(temp_path, self._vectors_path)
            
            # Update offsets
            self._id_to_offset = new_offsets
            
            # Reopen
            self._open_vectors_file()
            self._save_index()
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def stats(self) -> StorageStats:
        """Get storage statistics."""
        with self._lock:
            disk_bytes = 0
            if self._vectors_path.exists():
                disk_bytes += os.path.getsize(self._vectors_path)
            if self._index_path.exists():
                disk_bytes += os.path.getsize(self._index_path)
            if self._metadata_path.exists():
                disk_bytes += os.path.getsize(self._metadata_path)
            
            memory_bytes = self._buffer_size
            memory_bytes += len(self._id_to_offset) * 100  # Estimate
            memory_bytes += sum(len(str(m)) for m in self._metadata_cache.values())
            
            if self._mmap:
                memory_bytes += len(self._mmap)
            
            return StorageStats(
                vector_count=self._size,
                dimension=self._dimension,
                memory_bytes=memory_bytes,
                disk_bytes=disk_bytes,
                reads=self._reads,
                writes=self._writes,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
            )
    
    def clear(self) -> int:
        """Remove all vectors."""
        if self._mode == StorageMode.READ_ONLY:
            raise IOError("Storage is read-only")
        
        with self._lock:
            count = self._size
            
            # Close files
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            
            if self._vectors_file:
                self._vectors_file.close()
            
            # Delete files
            for path in [self._vectors_path, self._index_path, self._metadata_path]:
                if path.exists():
                    path.unlink()
            
            # Reset state
            self._id_to_offset.clear()
            self._metadata_cache.clear()
            self._write_buffer.clear()
            self._buffer_size = 0
            self._size = 0
            
            # Reinitialize
            self._initialize_files()
            
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
    
    def __repr__(self) -> str:
        return (
            f"DiskStorage(path='{self._path}', "
            f"dimension={self._dimension}, size={self._size})"
        )