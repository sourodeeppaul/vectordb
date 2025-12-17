"""
Serialization utilities for VectorDB storage.

Provides efficient serialization/deserialization for:
- Vectors (numpy arrays)
- Metadata (dictionaries)
- Index structures
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import struct
from typing import Dict, Any, List, Optional, BinaryIO
import json

# Try to import msgpack for efficient serialization
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


class VectorSerializer:
    """
    Efficient vector serialization.
    
    Supports:
    - Raw binary format (fastest)
    - Numpy .npy format (compatible)
    - Compressed format (smallest)
    """
    
    def __init__(
        self,
        dimension: int,
        dtype: np.dtype = np.float32,
        compressed: bool = False,
    ):
        self.dimension = dimension
        self.dtype = dtype
        self.compressed = compressed
        self.vector_size = dimension * np.dtype(dtype).itemsize
    
    def serialize(self, vector: NDArray) -> bytes:
        """Serialize a vector to bytes."""
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} != {self.dimension}"
            )
        
        # Ensure correct dtype
        if vector.dtype != self.dtype:
            vector = vector.astype(self.dtype)
        
        data = vector.tobytes()
        
        if self.compressed:
            import zlib
            data = zlib.compress(data, level=1)
        
        return data
    
    def deserialize(self, data: bytes) -> NDArray:
        """Deserialize bytes to vector."""
        if self.compressed:
            import zlib
            data = zlib.decompress(data)
        
        return np.frombuffer(data, dtype=self.dtype).copy()
    
    def serialize_batch(self, vectors: NDArray) -> bytes:
        """Serialize multiple vectors."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != {self.dimension}"
            )
        
        if vectors.dtype != self.dtype:
            vectors = vectors.astype(self.dtype)
        
        data = vectors.tobytes()
        
        if self.compressed:
            import zlib
            data = zlib.compress(data, level=1)
        
        return data
    
    def deserialize_batch(self, data: bytes, count: int) -> NDArray:
        """Deserialize bytes to multiple vectors."""
        if self.compressed:
            import zlib
            data = zlib.decompress(data)
        
        vectors = np.frombuffer(data, dtype=self.dtype)
        return vectors.reshape(count, self.dimension).copy()


def serialize_vector(vector: NDArray, compressed: bool = False) -> bytes:
    """
    Serialize a single vector.
    
    Format:
        - dimension (4 bytes, uint32)
        - dtype code (1 byte)
        - data (dimension * dtype_size bytes)
    """
    dtype_codes = {
        np.float32: 0,
        np.float64: 1,
        np.float16: 2,
    }
    
    dtype_code = dtype_codes.get(vector.dtype.type, 0)
    if dtype_code == 0 and vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    
    header = struct.pack('<IB', len(vector), dtype_code)
    data = vector.tobytes()
    
    if compressed:
        import zlib
        data = zlib.compress(data, level=1)
        header = struct.pack('<IBB', len(vector), dtype_code, 1)  # compression flag
    
    return header + data


def deserialize_vector(data: bytes) -> NDArray:
    """Deserialize bytes to vector."""
    dtype_map = {
        0: np.float32,
        1: np.float64,
        2: np.float16,
    }
    
    # Check header size
    if len(data) < 5:
        raise ValueError("Data too short for vector header")
    
    # Parse header
    if len(data) >= 6:
        dimension, dtype_code, compressed = struct.unpack('<IBB', data[:6])
        payload = data[6:]
    else:
        dimension, dtype_code = struct.unpack('<IB', data[:5])
        compressed = 0
        payload = data[5:]
    
    dtype = dtype_map.get(dtype_code, np.float32)
    
    if compressed:
        import zlib
        payload = zlib.decompress(payload)
    
    return np.frombuffer(payload, dtype=dtype).copy()


def serialize_metadata(metadata: Dict[str, Any]) -> bytes:
    """
    Serialize metadata dictionary.
    
    Uses msgpack if available, falls back to JSON.
    """
    if HAS_MSGPACK:
        return msgpack.packb(metadata, use_bin_type=True)
    else:
        return json.dumps(metadata).encode('utf-8')


def deserialize_metadata(data: bytes) -> Dict[str, Any]:
    """Deserialize metadata from bytes."""
    if not data:
        return {}
    
    if HAS_MSGPACK:
        try:
            return msgpack.unpackb(data, raw=False)
        except:
            pass
    
    # Fallback to JSON
    try:
        return json.loads(data.decode('utf-8'))
    except:
        return {}


class IndexSerializer:
    """Serialization for index structures."""
    
    @staticmethod
    def serialize_id_mapping(
        mapping: Dict[str, int]
    ) -> bytes:
        """Serialize ID to index mapping."""
        if HAS_MSGPACK:
            return msgpack.packb(mapping, use_bin_type=True)
        return json.dumps(mapping).encode('utf-8')
    
    @staticmethod
    def deserialize_id_mapping(data: bytes) -> Dict[str, int]:
        """Deserialize ID to index mapping."""
        if HAS_MSGPACK:
            try:
                return msgpack.unpackb(data, raw=False)
            except:
                pass
        return json.loads(data.decode('utf-8'))
    
    @staticmethod
    def serialize_neighbors(
        neighbors: Dict[int, List[str]]
    ) -> bytes:
        """Serialize neighbor lists (for HNSW)."""
        if HAS_MSGPACK:
            return msgpack.packb(neighbors, use_bin_type=True)
        return json.dumps(neighbors).encode('utf-8')
    
    @staticmethod
    def deserialize_neighbors(data: bytes) -> Dict[int, List[str]]:
        """Deserialize neighbor lists."""
        if HAS_MSGPACK:
            try:
                result = msgpack.unpackb(data, raw=False)
                # Convert string keys back to int
                return {int(k): v for k, v in result.items()}
            except:
                pass
        result = json.loads(data.decode('utf-8'))
        return {int(k): v for k, v in result.items()}


class StreamWriter:
    """
    Streaming writer for large files.
    
    Writes data incrementally to avoid memory issues.
    """
    
    def __init__(self, file: BinaryIO, dimension: int):
        self.file = file
        self.dimension = dimension
        self.serializer = VectorSerializer(dimension)
        self.count = 0
        self.vectors_start = 0
    
    def write_header(self, header: 'FileHeader') -> None:
        """Write file header."""
        from .format import FileHeader
        self.file.write(header.to_bytes())
        self.vectors_start = self.file.tell()
    
    def write_vector(self, vector: NDArray) -> int:
        """
        Write a vector and return its offset.
        """
        offset = self.file.tell()
        self.file.write(self.serializer.serialize(vector))
        self.count += 1
        return offset
    
    def write_vectors(self, vectors: NDArray) -> int:
        """Write multiple vectors and return start offset."""
        offset = self.file.tell()
        self.file.write(self.serializer.serialize_batch(vectors))
        self.count += len(vectors)
        return offset
    
    def write_metadata(self, metadata: Dict[str, Any]) -> tuple[int, int]:
        """
        Write metadata and return (offset, length).
        """
        offset = self.file.tell()
        data = serialize_metadata(metadata)
        self.file.write(data)
        return offset, len(data)
    
    def flush(self) -> None:
        """Flush to disk."""
        self.file.flush()


class StreamReader:
    """
    Streaming reader for large files.
    """
    
    def __init__(self, file: BinaryIO, header: 'FileHeader'):
        self.file = file
        self.header = header
        self.dimension = header.dimension
        self.serializer = VectorSerializer(header.dimension)
    
    def read_vector(self, offset: int) -> NDArray:
        """Read a vector from offset."""
        self.file.seek(offset)
        data = self.file.read(self.serializer.vector_size)
        return self.serializer.deserialize(data)
    
    def read_vectors(self, offset: int, count: int) -> NDArray:
        """Read multiple vectors from offset."""
        self.file.seek(offset)
        size = self.serializer.vector_size * count
        data = self.file.read(size)
        return self.serializer.deserialize_batch(data, count)
    
    def read_metadata(self, offset: int, length: int) -> Dict[str, Any]:
        """Read metadata from offset."""
        self.file.seek(offset)
        data = self.file.read(length)
        return deserialize_metadata(data)
    
    def read_all_vectors(self) -> NDArray:
        """Read all vectors."""
        return self.read_vectors(
            self.header.vectors_offset,
            self.header.vector_count
        )