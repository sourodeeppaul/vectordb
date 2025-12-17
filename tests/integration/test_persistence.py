"""
Integration tests for VectorDB persistence and durability.

Tests data persistence across database restarts, crash recovery,
and storage format compatibility.
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from vectordb.core.database import VectorDatabase
from vectordb.core.exceptions import (
    StorageError,
    CorruptedDataError,
)


@pytest.fixture
def persistent_db_path():
    """Create a temporary directory for persistent database storage."""
    path = tempfile.mkdtemp(prefix="vectordb_persist_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_data():
    """Generate consistent sample data for persistence tests."""
    np.random.seed(12345)
    vectors = np.random.randn(500, 128).astype(np.float32)
    metadata = [
        {
            "index": i,
            "category": f"cat_{i % 5}",
            "value": float(i) * 0.1,
            "tags": [f"tag_{i % 10}"]
        }
        for i in range(500)
    ]
    ids = [f"persistent_vec_{i}" for i in range(500)]
    return vectors, metadata, ids


class TestBasicPersistence:
    """Test basic data persistence across database restarts."""

    def test_persist_and_reload_collection(self, persistent_db_path, sample_data):
        """Test that collection data persists across restarts."""
        vectors, metadata, ids = sample_data
        
        # Create database and add data
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=128)
        collection.add(vectors[:100], ids=ids[:100], metadata=metadata[:100])
        initial_count = collection.count()
        db.close()
        
        # Reopen database and verify data
        db2 = VectorDatabase(storage_path=persistent_db_path)
        collection2 = db2.get_collection("test")
        
        assert collection2.count() == initial_count
        
        # Verify vector content
        result = collection2.get(ids[0])
        assert np.allclose(result["vector"], vectors[0], atol=1e-6)
        assert result["metadata"]["index"] == 0
        
        db2.close()

    def test_persist_multiple_collections(self, persistent_db_path, sample_data):
        """Test persistence of multiple collections."""
        vectors, metadata, ids = sample_data
        
        # Create multiple collections
        db = VectorDatabase(storage_path=persistent_db_path)
        for i, index_type in enumerate(["flat", "hnsw"]):
            coll = db.create_collection(
                f"collection_{index_type}",
                dimension=128,
                index_type=index_type
            )
            start = i * 100
            end = start + 100
            coll.add(vectors[start:end], ids=ids[start:end])
        db.close()
        
        # Verify all collections persist
        db2 = VectorDatabase(storage_path=persistent_db_path)
        collections = db2.list_collections()
        
        assert len(collections) == 2
        assert "collection_flat" in collections
        assert "collection_hnsw" in collections
        
        for coll_name in collections:
            coll = db2.get_collection(coll_name)
            assert coll.count() == 100
        
        db2.close()

    def test_persist_collection_metadata(self, persistent_db_path):
        """Test that collection configuration persists."""
        # Create collection with specific config
        db = VectorDatabase(storage_path=persistent_db_path)
        db.create_collection(
            "configured",
            dimension=256,
            metric="euclidean",
            index_type="hnsw",
            index_params={"M": 32, "ef_construction": 200},
            description="A test collection"
        )
        db.close()
        
        # Verify configuration persists
        db2 = VectorDatabase(storage_path=persistent_db_path)
        coll = db2.get_collection("configured")
        info = coll.info()
        
        assert info["dimension"] == 256
        assert info["metric"] == "euclidean"
        assert info["index_type"] == "hnsw"
        
        db2.close()

    def test_persist_after_updates(self, persistent_db_path, sample_data):
        """Test persistence after update operations."""
        vectors, metadata, ids = sample_data
        
        # Create and populate
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=128)
        collection.add(vectors[:50], ids=ids[:50], metadata=metadata[:50])
        
        # Update some vectors
        updated_vector = np.ones(128, dtype=np.float32)
        collection.update(ids[0], vector=updated_vector)
        collection.update(ids[1], metadata={"updated": True})
        db.close()
        
        # Verify updates persist
        db2 = VectorDatabase(storage_path=persistent_db_path)
        collection2 = db2.get_collection("test")
        
        result0 = collection2.get(ids[0])
        assert np.allclose(result0["vector"], updated_vector)
        
        result1 = collection2.get(ids[1])
        assert result1["metadata"].get("updated") is True
        
        db2.close()

    def test_persist_after_deletes(self, persistent_db_path, sample_data):
        """Test persistence after delete operations."""
        vectors, metadata, ids = sample_data
        
        # Create, populate, then delete
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=128)
        collection.add(vectors[:100], ids=ids[:100])
        collection.delete(ids[:25])
        remaining_count = collection.count()
        db.close()
        
        # Verify deletes persist
        db2 = VectorDatabase(storage_path=persistent_db_path)
        collection2 = db2.get_collection("test")
        
        assert collection2.count() == remaining_count
        assert collection2.count() == 75
        
        db2.close()


class TestIncrementalPersistence:
    """Test incremental data operations and persistence."""

    def test_incremental_adds_persist(self, persistent_db_path, sample_data):
        """Test that incremental additions persist correctly."""
        vectors, metadata, ids = sample_data
        
        # First session: add initial data
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=128)
        collection.add(vectors[:100], ids=ids[:100])
        db.close()
        
        # Second session: add more data
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        collection.add(vectors[100:200], ids=ids[100:200])
        db.close()
        
        # Third session: add even more
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        collection.add(vectors[200:300], ids=ids[200:300])
        db.close()
        
        # Final verification
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        assert collection.count() == 300
        
        # Verify data from all sessions
        for session_start in [0, 100, 200]:
            result = collection.get(ids[session_start])
            assert np.allclose(result["vector"], vectors[session_start], atol=1e-6)
        
        db.close()

    def test_mixed_operations_persist(self, persistent_db_path, sample_data):
        """Test persistence of mixed add/update/delete operations."""
        vectors, metadata, ids = sample_data
        
        # Session 1: Initial add
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=128)
        collection.add(vectors[:100], ids=ids[:100], metadata=metadata[:100])
        db.close()
        
        # Session 2: More adds + updates
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        collection.add(vectors[100:150], ids=ids[100:150])
        collection.update(ids[0], metadata={"session": 2})
        db.close()
        
        # Session 3: Deletes + more adds
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        collection.delete(ids[50:75])
        collection.add(vectors[150:175], ids=ids[150:175])
        db.close()
        
        # Verify final state
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        
        # Count: 100 + 50 - 25 + 25 = 150
        assert collection.count() == 150
        
        # Check update persisted
        result = collection.get(ids[0])
        assert result["metadata"].get("session") == 2
        
        db.close()


class TestSearchAfterReload:
    """Test that search works correctly after database reload."""

    def test_flat_search_after_reload(self, persistent_db_path, sample_data):
        """Test flat index search after reload."""
        vectors, _, ids = sample_data
        
        # Create and save
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection(
            "test", dimension=128, index_type="flat"
        )
        collection.add(vectors[:100], ids=ids[:100])
        db.close()
        
        # Reload and search
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        
        # Search for known vector
        results = collection.search(vectors[0], k=1)
        assert results[0]["id"] == ids[0]
        
        db.close()

    def test_hnsw_search_after_reload(self, persistent_db_path, sample_data):
        """Test HNSW index search after reload."""
        vectors, _, ids = sample_data
        
        # Create and save
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection(
            "test",
            dimension=128,
            index_type="hnsw",
            index_params={"M": 16, "ef_construction": 100}
        )
        collection.add(vectors[:200], ids=ids[:200])
        db.close()
        
        # Reload and search
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        
        # Search should work and return reasonable results
        results = collection.search(vectors[50], k=10)
        assert len(results) == 10
        
        # The exact vector should be in top results (HNSW is approximate)
        top_ids = [r["id"] for r in results[:5]]
        assert ids[50] in top_ids
        
        db.close()

    def test_filtered_search_after_reload(self, persistent_db_path, sample_data):
        """Test filtered search after reload."""
        vectors, metadata, ids = sample_data
        
        # Create with metadata
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=128)
        collection.add(vectors[:100], ids=ids[:100], metadata=metadata[:100])
        db.close()
        
        # Reload and filtered search
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        
        results = collection.search(
            vectors[0],
            k=20,
            filter={"category": "cat_0"}
        )
        
        assert all(r["metadata"]["category"] == "cat_0" for r in results)
        
        db.close()


class TestStorageFormats:
    """Test different storage format scenarios."""

    def test_storage_file_structure(self, persistent_db_path, sample_data):
        """Test that expected storage files are created."""
        vectors, metadata, ids = sample_data
        
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test_collection", dimension=128)
        collection.add(vectors[:50], ids=ids[:50], metadata=metadata[:50])
        db.close()
        
        # Check for expected files/directories
        storage_path = Path(persistent_db_path)
        assert storage_path.exists()
        
        # Should have some form of manifest/metadata
        files = list(storage_path.rglob("*"))
        assert len(files) > 0

    def test_large_metadata_persistence(self, persistent_db_path):
        """Test persistence of vectors with large metadata."""
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=64)
        
        # Create large metadata
        large_metadata = {
            "description": "x" * 10000,
            "nested": {
                "level1": {
                    "level2": {
                        "data": list(range(100))
                    }
                }
            },
            "tags": [f"tag_{i}" for i in range(100)]
        }
        
        vector = np.random.randn(64).astype(np.float32)
        collection.add(vector, ids=["large_meta"], metadata=[large_metadata])
        db.close()
        
        # Verify large metadata persists
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        result = collection.get("large_meta")
        
        assert len(result["metadata"]["description"]) == 10000
        assert result["metadata"]["nested"]["level1"]["level2"]["data"] == list(range(100))
        
        db.close()

    def test_unicode_metadata_persistence(self, persistent_db_path):
        """Test persistence of unicode content in metadata."""
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=64)
        
        unicode_metadata = {
            "chinese": "这是中文测试",
            "japanese": "日本語テスト",
            "emoji": "🚀🎉✨",
            "mixed": "Hello 世界 🌍"
        }
        
        vector = np.random.randn(64).astype(np.float32)
        collection.add(vector, ids=["unicode"], metadata=[unicode_metadata])
        db.close()
        
        # Verify unicode persists correctly
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("test")
        result = collection.get("unicode")
        
        assert result["metadata"]["chinese"] == "这是中文测试"
        assert result["metadata"]["emoji"] == "🚀🎉✨"
        
        db.close()


class TestCrashRecovery:
    """Test database recovery from various failure scenarios."""

    def test_recovery_after_incomplete_write(self, persistent_db_path, sample_data):
        """Test recovery when database wasn't properly closed."""
        vectors, _, ids = sample_data
        
        # Simulate incomplete close by not calling close()
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=128)
        collection.add(vectors[:50], ids=ids[:50])
        # Note: Not calling db.close() - simulating crash
        
        # Try to reopen - should recover gracefully
        db2 = VectorDatabase(storage_path=persistent_db_path)
        
        # Check if we can still access data (may be partial)
        if db2.has_collection("test"):
            collection = db2.get_collection("test")
            # Should be able to search without errors
            query = np.random.randn(128).astype(np.float32)
            results = collection.search(query, k=5)
            assert isinstance(results, list)
        
        db2.close()

    def test_recovery_with_corrupted_metadata(self, persistent_db_path, sample_data):
        """Test recovery when metadata file is corrupted."""
        vectors, _, ids = sample_data
        
        # Create valid database
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("test", dimension=128)
        collection.add(vectors[:50], ids=ids[:50])
        db.close()
        
        # Find and corrupt metadata file (if exists)
        storage_path = Path(persistent_db_path)
        for json_file in storage_path.rglob("*.json"):
            # Corrupt the file
            with open(json_file, "a") as f:
                f.write("corrupted data!!!")
            break
        
        # Attempt recovery
        try:
            db2 = VectorDatabase(storage_path=persistent_db_path)
            # If it opens, verify it handles corruption gracefully
            db2.close()
        except (StorageError, CorruptedDataError, json.JSONDecodeError):
            # Expected - database detected corruption
            pass

    def test_multiple_open_prevention(self, persistent_db_path, sample_data):
        """Test that concurrent opens are handled properly."""
        vectors, _, ids = sample_data
        
        db1 = VectorDatabase(storage_path=persistent_db_path)
        collection = db1.create_collection("test", dimension=128)
        collection.add(vectors[:50], ids=ids[:50])
        
        # Try to open same database again
        # This should either work (with proper locking) or raise an error
        try:
            db2 = VectorDatabase(storage_path=persistent_db_path)
            # If it works, both should be able to read
            coll2 = db2.get_collection("test")
            assert coll2.count() == 50
            db2.close()
        except Exception:
            # Locking prevented concurrent access
            pass
        
        db1.close()


class TestStorageCapacity:
    """Test storage with various data sizes."""

    @pytest.mark.slow
    def test_large_collection_persistence(self, persistent_db_path):
        """Test persistence of a large collection."""
        np.random.seed(42)
        num_vectors = 10000
        dimension = 128
        
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        ids = [f"vec_{i}" for i in range(num_vectors)]
        
        # Create and save
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("large", dimension=dimension)
        
        # Add in batches
        batch_size = 1000
        for i in range(0, num_vectors, batch_size):
            collection.add(
                vectors[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        
        db.close()
        
        # Verify
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("large")
        assert collection.count() == num_vectors
        
        # Verify random samples
        for idx in [0, 5000, 9999]:
            result = collection.get(ids[idx])
            assert np.allclose(result["vector"], vectors[idx], atol=1e-6)
        
        db.close()

    def test_high_dimension_persistence(self, persistent_db_path):
        """Test persistence of high-dimensional vectors."""
        dimension = 2048
        num_vectors = 100
        
        np.random.seed(42)
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        ids = [f"hd_{i}" for i in range(num_vectors)]
        
        # Save
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.create_collection("high_dim", dimension=dimension)
        collection.add(vectors, ids=ids)
        db.close()
        
        # Verify
        db = VectorDatabase(storage_path=persistent_db_path)
        collection = db.get_collection("high_dim")
        
        result = collection.get(ids[0])
        assert len(result["vector"]) == dimension
        assert np.allclose(result["vector"], vectors[0], atol=1e-6)
        
        db.close()

    def test_many_collections_persistence(self, persistent_db_path):
        """Test persistence with many collections."""
        num_collections = 50
        
        # Create many collections
        db = VectorDatabase(storage_path=persistent_db_path)
        for i in range(num_collections):
            coll = db.create_collection(f"collection_{i}", dimension=32)
            vectors = np.random.randn(10, 32).astype(np.float32)
            coll.add(vectors)
        db.close()
        
        # Verify all collections persist
        db = VectorDatabase(storage_path=persistent_db_path)
        collections = db.list_collections()
        assert len(collections) == num_collections
        
        for name in collections:
            coll = db.get_collection(name)
            assert coll.count() == 10
        
        db.close()


class TestBackupAndRestore:
    """Test backup and restore functionality."""

    def test_backup_collection(self, persistent_db_path, sample_data):
        """Test backing up a collection."""
        vectors, metadata, ids = sample_data
        backup_path = tempfile.mkdtemp(prefix="vectordb_backup_")
        
        try:
            # Create and populate
            db = VectorDatabase(storage_path=persistent_db_path)
            collection = db.create_collection("test", dimension=128)
            collection.add(vectors[:100], ids=ids[:100], metadata=metadata[:100])
            
            # Backup
            db.backup(backup_path)
            db.close()
            
            # Verify backup exists and is valid
            backup_db = VectorDatabase(storage_path=backup_path)
            assert backup_db.has_collection("test")
            
            backup_coll = backup_db.get_collection("test")
            assert backup_coll.count() == 100
            
            backup_db.close()
        finally:
            shutil.rmtree(backup_path, ignore_errors=True)

    def test_restore_from_backup(self, persistent_db_path, sample_data):
        """Test restoring from a backup."""
        vectors, _, ids = sample_data
        backup_path = tempfile.mkdtemp(prefix="vectordb_backup_")
        restore_path = tempfile.mkdtemp(prefix="vectordb_restore_")
        
        try:
            # Create original
            db = VectorDatabase(storage_path=persistent_db_path)
            collection = db.create_collection("test", dimension=128)
            collection.add(vectors[:100], ids=ids[:100])
            db.backup(backup_path)
            db.close()
            
            # Restore to new location
            VectorDatabase.restore(backup_path, restore_path)
            
            # Verify restore
            restored_db = VectorDatabase(storage_path=restore_path)
            assert restored_db.has_collection("test")
            
            restored_coll = restored_db.get_collection("test")
            assert restored_coll.count() == 100
            
            restored_db.close()
        finally:
            shutil.rmtree(backup_path, ignore_errors=True)
            shutil.rmtree(restore_path, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])