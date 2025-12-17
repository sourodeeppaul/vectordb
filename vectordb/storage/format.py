"""
File format definitions for VectorDB storage.

Defines the binary format used for persistent storage,
including headers, sections, and checksums.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import IntFlag
import hashlib


# Magic number: "VECTRDB\x00"
MAGIC_NUMBER = b'VECTRDB\x00'

# File format version
VERSION = 1

# Section alignment (for memory mapping efficiency)
ALIGNMENT = 4096  # 4KB pages


class FileFlags(IntFlag):
    """File format flags."""
    NONE = 0
    COMPRESSED = 1 << 0      # Vectors are compressed
    NORMALIZED = 1 << 1      # Vectors are normalized
    HAS_METADATA = 1 << 2    # File contains metadata
    HAS_INDEX = 1 << 3       # File contains index structure
    MMAP_OPTIMIZED = 1 << 4  # Optimized for memory mapping


@dataclass
class FileHeader:
    """
    File header structure (64 bytes).
    
    Layout:
        0-7:   Magic number (8 bytes)
        8-11:  Version (4 bytes, uint32)
        12-15: Flags (4 bytes, uint32)
        16-19: Dimension (4 bytes, uint32)
        20-27: Vector count (8 bytes, uint64)
        28-35: Vectors section offset (8 bytes, uint64)
        36-43: Index section offset (8 bytes, uint64)
        44-51: Metadata section offset (8 bytes, uint64)
        52-63: Reserved (12 bytes)
    """
    
    magic: bytes = MAGIC_NUMBER
    version: int = VERSION
    flags: int = FileFlags.NONE
    dimension: int = 0
    vector_count: int = 0
    vectors_offset: int = 64  # Right after header
    index_offset: int = 0
    metadata_offset: int = 0
    
    # Format string for struct packing
    FORMAT = '<8sIIIQQQQ12x'
    SIZE = 64
    
    def validate(self) -> bool:
        """Validate header."""
        if self.magic != MAGIC_NUMBER:
            raise ValueError(f"Invalid magic number: {self.magic}")
        if self.version > VERSION:
            raise ValueError(f"Unsupported version: {self.version}")
        if self.dimension <= 0:
            raise ValueError(f"Invalid dimension: {self.dimension}")
        return True
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            self.FORMAT,
            self.magic,
            self.version,
            self.flags,
            self.dimension,
            self.vector_count,
            self.vectors_offset,
            self.index_offset,
            self.metadata_offset,
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'FileHeader':
        """Deserialize header from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Header too short: {len(data)} < {cls.SIZE}")
        
        unpacked = struct.unpack(cls.FORMAT, data[:cls.SIZE])
        
        header = cls(
            magic=unpacked[0],
            version=unpacked[1],
            flags=unpacked[2],
            dimension=unpacked[3],
            vector_count=unpacked[4],
            vectors_offset=unpacked[5],
            index_offset=unpacked[6],
            metadata_offset=unpacked[7],
        )
        
        header.validate()
        return header
    
    def __repr__(self) -> str:
        return (
            f"FileHeader(version={self.version}, dimension={self.dimension}, "
            f"count={self.vector_count}, flags={self.flags})"
        )


@dataclass
class IndexEntry:
    """
    Entry in the index section.
    
    Layout:
        0-3:   ID length (4 bytes, uint32)
        4-N:   ID string (variable)
        N-N+8: Vector offset (8 bytes, uint64)
        N+8-N+16: Metadata offset (8 bytes, uint64)
        N+16-N+20: Metadata length (4 bytes, uint32)
    """
    
    id: str
    vector_offset: int
    metadata_offset: int
    metadata_length: int
    
    def to_bytes(self) -> bytes:
        """Serialize entry to bytes."""
        id_bytes = self.id.encode('utf-8')
        
        return struct.pack(
            f'<I{len(id_bytes)}sQQI',
            len(id_bytes),
            id_bytes,
            self.vector_offset,
            self.metadata_offset,
            self.metadata_length,
        )
    
    @classmethod
    def from_buffer(cls, buffer: bytes, offset: int) -> tuple['IndexEntry', int]:
        """
        Deserialize entry from buffer at offset.
        
        Returns:
            Tuple of (entry, new_offset)
        """
        # Read ID length
        id_len = struct.unpack_from('<I', buffer, offset)[0]
        offset += 4
        
        # Read ID
        id_bytes = buffer[offset:offset + id_len]
        id_str = id_bytes.decode('utf-8')
        offset += id_len
        
        # Read offsets
        vector_offset, metadata_offset, metadata_length = struct.unpack_from(
            '<QQI', buffer, offset
        )
        offset += 20
        
        entry = cls(
            id=id_str,
            vector_offset=vector_offset,
            metadata_offset=metadata_offset,
            metadata_length=metadata_length,
        )
        
        return entry, offset


@dataclass
class FileFooter:
    """
    File footer structure (32 bytes).
    
    Layout:
        0-7:   Checksum (8 bytes, uint64)
        8-31:  Reserved (24 bytes)
    """
    
    checksum: int = 0
    
    FORMAT = '<Q24x'
    SIZE = 32
    
    def to_bytes(self) -> bytes:
        """Serialize footer to bytes."""
        return struct.pack(self.FORMAT, self.checksum)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'FileFooter':
        """Deserialize footer from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Footer too short: {len(data)} < {cls.SIZE}")
        
        checksum = struct.unpack(cls.FORMAT, data[:cls.SIZE])[0]
        return cls(checksum=checksum)


class FileFormat:
    """
    Helper class for file format operations.
    """
    
    @staticmethod
    def align_offset(offset: int, alignment: int = ALIGNMENT) -> int:
        """Align offset to boundary."""
        if offset % alignment == 0:
            return offset
        return offset + (alignment - offset % alignment)
    
    @staticmethod
    def compute_checksum(data: bytes) -> int:
        """Compute checksum for data."""
        # Use xxhash if available, fallback to simple hash
        try:
            import xxhash
            return xxhash.xxh64(data).intdigest()
        except ImportError:
            # Fallback to MD5 truncated to 64 bits
            md5 = hashlib.md5(data).digest()
            return struct.unpack('<Q', md5[:8])[0]
    
    @staticmethod
    def vector_size(dimension: int) -> int:
        """Calculate size of a single vector in bytes."""
        return dimension * 4  # float32
    
    @staticmethod
    def vectors_section_size(dimension: int, count: int) -> int:
        """Calculate size of vectors section."""
        return dimension * 4 * count