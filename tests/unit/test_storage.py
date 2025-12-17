"""
Unit tests for storage backends.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import tempfile
import shutil
from pathlib import Path

from vectordb.storage import (
    MemoryStorage,
    DiskStorage,
    StorageStats,
)
from vectordb.storage.format import FileHeader, FileFlags, MAGIC_NUMBER
from vectordb.storage.serialization import (
    VectorSerializer,
    serialize_vector,
    deserialize_vector,
    serialize_metadata,
    deserialize_metadata,
)


class TestVectorSerializer:
    """Test vector serialization."""
    
    def test_serialize_deserialize(self):
        """Test round-trip serialization."""
        serializer = VectorSerializer(dimension=128)
        
        vector = np.random.randn(128).astype(np.float32)
        
        data = serializer.serialize(vector)
        restored = serializer.deserialize(data)
        
        assert_array_almost_equal(vector, restored)
    
    def test_serialize_batch(self):
        """Test batch serialization."""
        serializer = VectorSerializer(dimension=64)
        
        vectors = np.random.randn(100, 64).astype(np.float32)
        
        data = serializer.serialize_batch(vectors)
        restored = serializer.deserialize_batch(data, 100)
        
        assert_array_almost_equal(vectors, restored)
    
    def test_wrong_dimension(self):
        """Test dimension mismatch."""
        serializer = VectorSerializer(dimension=128)
        
        vector = np.random.randn(64).astype(np.float32)
        
        with pytest.raises(ValueError):
            serializer.serialize(vector)


class TestMetadataSerialization:
    """Test metadata serialization."""
    
    def test_serialize_deserialize(self):
        """Test round-trip serialization."""
        metadata = {
            "name": "test",
            "value": 42,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }
        
        data = serialize_metadata(metadata)
        restored = deserialize_metadata(data)
        
        assert restored == metadata
    
    def test_empty_metadata(self):
        """Test empty metadata."""
        data = serialize_metadata({})
        restored = deserialize_metadata(data)
        
        assert restored == {}


class TestFileHeader:
    """Test file header."""
    
    def test_serialize_deserialize(self):
        """Test round-trip serialization."""
        header = FileHeader(
            dimension=128,
            vector_count=1000,
            flags=FileFlags.HAS_METADATA,
        )
        
        data = header.to_bytes()
        restored = FileHeader.from_bytes(data)
        
        assert restored.dimension == 128
        assert restored.vector_count == 1000
        assert restored.flags == FileFlags.HAS_METADATA
    
    def test_header_size(self):
        """Test header size."""
        header = FileHeader(dimension=128)
        data = header.to_bytes()
        
        assert len(data) == FileHeader.SIZE == 64
    
    def test_invalid_magic(self):
        """Test invalid magic number."""
        header = FileHeader(dimension=128)
        data = bytearray(header.to_bytes())
        data[0:8] = b'INVALID\x00'
        
        with pytest.raises(ValueError, match="magic"):
            FileHeader.from_bytes(bytes(data))


class TestMemoryStorage:
    """Test memory storage."""
    
    @pytest.fixture
    def storage(self):
        return MemoryStorage(dimension=128)
    
    @pytest.fixture
    def vector(self):
        return np.random.randn(128).astype(np.float32)
    
    def test_put_get(self, storage, vector):
        """Test basic put and get."""
        storage.put("id1", vector, {"key": "value"})
        
        result = storage.get("id1")
        
        assert result is not None
        vec, meta = result
        assert_array_almost_equal(vec, vector)
        assert meta["key"] == "value"
    
    def test_get_not_found(self, storage):
        """Test getting non-existent ID."""
        result = storage.get("nonexistent")
        assert result is None
    
    def test_delete(self, storage, vector):
        """Test delete."""
        storage.put("id1", vector)
        
        result = storage.delete("id1")
        
        assert result is True
        assert storage.get("id1") is None
        assert storage.size == 0
    
    def test_delete_not_found(self, storage):
        """Test deleting non-existent ID."""
        result = storage.delete("nonexistent")
        assert result is False
    
    def test_contains(self, storage, vector):
        """Test contains check."""
        storage.put("id1", vector)
        
        assert storage.contains("id1")
        assert "id1" in storage
        assert not storage.contains("id2")
    
    def test_update(self, storage, vector):
        """Test update."""
        storage.put("id1", vector, {"v": 1})
        
        new_vector = np.random.randn(128).astype(np.float32)
        storage.update("id1", vector=new_vector, metadata={"v": 2})
        
        result = storage.get("id1")
        vec, meta = result
        
        assert_array_almost_equal(vec, new_vector)
        assert meta["v"] == 2
    
    def test_update_not_found(self, storage, vector):
        """Test updating non-existent ID."""
        result = storage.update("nonexistent", vector=vector)
        assert result is False
    
    def test_put_batch(self, storage):
        """Test batch put."""
        ids = [f"id{i}" for i in range(100)]
        vectors = np.random.randn(100, 128).astype(np.float32)
        metadata = [{"i": i} for i in range(100)]
        
        count = storage.put_batch(ids, vectors, metadata)
        
        assert count == 100
        assert storage.size == 100
    
    def test_get_batch(self, storage):
        """Test batch get."""
        ids = [f"id{i}" for i in range(10)]
        vectors = np.random.randn(10, 128).astype(np.float32)
        storage.put_batch(ids, vectors)
        
        results = storage.get_batch(["id0", "id5", "nonexistent"])
        
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None
    
    def test_get_all_vectors(self, storage):
        """Test getting all vectors."""
        vectors = np.random.randn(50, 128).astype(np.float32)
        ids = [f"id{i}" for i in range(50)]
        storage.put_batch(ids, vectors)
        
        all_vectors = storage.get_all_vectors()
        
        assert all_vectors.shape == (50, 128)
    
    def test_get_all_ids(self, storage, vector):
        """Test getting all IDs."""
        for i in range(10):
            storage.put(f"id{i}", vector)
        
        ids = storage.get_all_ids()
        
        assert len(ids) == 10
        assert "id0" in ids
    
    def test_iteration(self, storage, vector):
        """Test iteration."""
        for i in range(10):
            storage.put(f"id{i}", vector)
        
        count = 0
        for id, vec, meta in storage.iter_vectors():
            assert len(vec) == 128
            count += 1
        
        assert count == 10
    
    def test_clear(self, storage, vector):
        """Test clear."""
        for i in range(10):
            storage.put(f"id{i}", vector)
        
        count = storage.clear()
        
        assert count == 10
        assert storage.size == 0
    
    def test_stats(self, storage, vector):
        """Test statistics."""
        for i in range(10):
            storage.put(f"id{i}", vector)
        
        stats = storage.stats()
        
        assert stats.vector_count == 10
        assert stats.dimension == 128
        assert stats.memory_bytes > 0
    
    def test_capacity_expansion(self, storage):
        """Test automatic capacity expansion."""
        storage = MemoryStorage(dimension=128, initial_capacity=10)
        
        vectors = np.random.randn(100, 128).astype(np.float32)
        ids = [f"id{i}" for i in range(100)]
        storage.put_batch(ids, vectors)
        
        assert storage.size == 100


class TestDiskStorage:
    """Test disk storage."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)
    
    @pytest.fixture
    def storage(self, temp_dir):
        """Create disk storage."""
        return DiskStorage(temp_dir, dimension=128)
    
    @pytest.fixture
    def vector(self):
        return np.random.randn(128).astype(np.float32)
    
    def test_put_get(self, storage, vector):
        """Test basic put and get."""
        storage.put("id1", vector, {"key": "value"})
        storage.flush()
        
        result = storage.get("id1")
        
        assert result is not None
        vec, meta = result
        assert_array_almost_equal(vec, vector)
        assert meta["key"] == "value"
    
    def test_persistence(self, temp_dir, vector):
        """Test data persists across instances."""
        # Write data
        storage1 = DiskStorage(temp_dir, dimension=128)
        storage1.put("id1", vector, {"key": "value"})
        storage1.flush()
        storage1.close()
        
        # Read with new instance
        storage2 = DiskStorage(temp_dir, dimension=128)
        result = storage2.get("id1")
        storage2.close()
        
        assert result is not None
        vec, meta = result
        assert_array_almost_equal(vec, vector)
        assert meta["key"] == "value"
    
    def test_delete(self, storage, vector):
        """Test delete."""
        storage.put("id1", vector)
        storage.flush()
        
        result = storage.delete("id1")
        
        assert result is True
        assert storage.get("id1") is None
    
    def test_batch_operations(self, storage):
        """Test batch put and get."""
        ids = [f"id{i}" for i in range(100)]
        vectors = np.random.randn(100, 128).astype(np.float32)
        
        storage.put_batch(ids, vectors)
        storage.flush()
        
        assert storage.size == 100
        
        # Get batch
        results = storage.get_batch(["id0", "id50", "id99"])
        assert all(r is not None for r in results)
    
    def test_context_manager(self, temp_dir, vector):
        """Test context manager."""
        with DiskStorage(temp_dir, dimension=128) as storage:
            storage.put("id1", vector)
        
        # Should be flushed and closed
        storage2 = DiskStorage(temp_dir, dimension=128)
        result = storage2.get("id1")
        storage2.close()
        
        assert result is not None
    
    def test_stats(self, storage, vector):
        """Test statistics."""
        for i in range(10):
            storage.put(f"id{i}", vector)
        storage.flush()
        
        stats = storage.stats()
        
        assert stats.vector_count == 10
        assert stats.dimension == 128
        assert stats.disk_bytes > 0
    
    def test_clear(self, storage, vector):
        """Test clear."""
        for i in range(10):
            storage.put(f"id{i}", vector)
        storage.flush()
        
        count = storage.clear()
        
        assert count == 10
        assert storage.size == 0
    
    def test_compact(self, storage, vector):
        """Test compaction."""
        # Add vectors
        for i in range(100):
            storage.put(f"id{i}", vector)
        storage.flush()
        
        # Delete half
        for i in range(0, 100, 2):
            storage.delete(f"id{i}")
        
        # Compact
        storage.compact()
        
        # Check remaining
        assert storage.size == 50
        for i in range(1, 100, 2):
            assert storage.get(f"id{i}") is not None
    
    def test_mmap_access(self, temp_dir, vector):
        """Test memory-mapped access."""
        # Write many vectors
        storage = DiskStorage(temp_dir, dimension=128, use_mmap=True)
        
        for i in range(1000):
            storage.put(f"id{i}", np.random.randn(128).astype(np.float32))
        storage.flush()
        
        # Random access should use mmap
        for _ in range(100):
            i = np.random.randint(1000)
            result = storage.get(f"id{i}")
            assert result is not None
        
        storage.close()


class TestDiskStorageLarge:
    """Test disk storage with larger datasets."""
    
    @pytest.fixture
    def temp_dir(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)
    
    @pytest.mark.slow
    def test_large_dataset(self, temp_dir):
        """Test with larger dataset."""
        storage = DiskStorage(temp_dir, dimension=128)
        
        n_vectors = 10000
        batch_size = 1000
        
        # Add in batches
        for start in range(0, n_vectors, batch_size):
            end = min(start + batch_size, n_vectors)
            ids = [f"vec{i}" for i in range(start, end)]
            vectors = np.random.randn(end - start, 128).astype(np.float32)
            storage.put_batch(ids, vectors)
        
        storage.flush()
        
        assert storage.size == n_vectors
        
        # Random access
        for _ in range(100):
            i = np.random.randint(n_vectors)
            result = storage.get(f"vec{i}")
            assert result is not None
        
        stats = storage.stats()
        print(f"\nLarge dataset stats: {stats.to_dict()}")
        
        storage.close()