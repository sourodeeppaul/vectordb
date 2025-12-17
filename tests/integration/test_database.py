"""
Integration tests for the VectorDB database operations.

Tests full database lifecycle including creation, operations,
and cleanup across all index types.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from vectordb.core.database import VectorDatabase
from vectordb.core.collection import Collection
from vectordb.core.exceptions import (
    CollectionNotFoundError,
    CollectionExistsError,
    DimensionMismatchError,
    VectorNotFoundError,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for database storage."""
    path = tempfile.mkdtemp(prefix="vectordb_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def database(temp_db_path):
    """Create a database instance for testing."""
    db = VectorDatabase(storage_path=temp_db_path)
    yield db
    db.close()


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)
    return np.random.randn(1000, 128).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    categories = ["tech", "science", "art", "sports", "music"]
    return [
        {
            "id": i,
            "category": categories[i % len(categories)],
            "score": float(i) / 100,
            "tags": [f"tag_{i % 10}", f"group_{i % 5}"],
            "active": i % 2 == 0,
        }
        for i in range(1000)
    ]


class TestDatabaseLifecycle:
    """Test database creation, configuration, and shutdown."""

    def test_database_creation(self, temp_db_path):
        """Test basic database creation."""
        db = VectorDatabase(storage_path=temp_db_path)
        assert db is not None
        assert db.storage_path == Path(temp_db_path)
        db.close()

    def test_database_creation_in_memory(self):
        """Test in-memory database creation."""
        db = VectorDatabase(storage_path=None)
        assert db is not None
        db.close()

    def test_database_creates_directory(self, temp_db_path):
        """Test that database creates storage directory."""
        nested_path = Path(temp_db_path) / "nested" / "db"
        db = VectorDatabase(storage_path=str(nested_path))
        assert nested_path.exists()
        db.close()

    def test_database_context_manager(self, temp_db_path):
        """Test database as context manager."""
        with VectorDatabase(storage_path=temp_db_path) as db:
            db.create_collection("test", dimension=64)
            assert db.has_collection("test")

    def test_database_double_close(self, temp_db_path):
        """Test that double close doesn't raise error."""
        db = VectorDatabase(storage_path=temp_db_path)
        db.close()
        db.close()  # Should not raise


class TestCollectionManagement:
    """Test collection CRUD operations."""

    def test_create_collection(self, database):
        """Test basic collection creation."""
        collection = database.create_collection(
            name="test_collection",
            dimension=128,
            metric="cosine",
        )
        assert collection is not None
        assert collection.name == "test_collection"
        assert collection.dimension == 128

    def test_create_collection_all_metrics(self, database):
        """Test collection creation with all metric types."""
        metrics = ["cosine", "euclidean", "dot", "manhattan"]
        for i, metric in enumerate(metrics):
            collection = database.create_collection(
                name=f"collection_{metric}",
                dimension=64,
                metric=metric,
            )
            assert collection.metric == metric

    def test_create_collection_all_index_types(self, database):
        """Test collection creation with all index types."""
        index_types = ["flat", "ivf", "hnsw"]
        for index_type in index_types:
            collection = database.create_collection(
                name=f"collection_{index_type}",
                dimension=64,
                index_type=index_type,
            )
            assert collection is not None

    def test_create_duplicate_collection(self, database):
        """Test that creating duplicate collection raises error."""
        database.create_collection("test", dimension=64)
        with pytest.raises(CollectionExistsError):
            database.create_collection("test", dimension=64)

    def test_get_collection(self, database):
        """Test getting an existing collection."""
        database.create_collection("test", dimension=64)
        collection = database.get_collection("test")
        assert collection is not None
        assert collection.name == "test"

    def test_get_nonexistent_collection(self, database):
        """Test getting a collection that doesn't exist."""
        with pytest.raises(CollectionNotFoundError):
            database.get_collection("nonexistent")

    def test_has_collection(self, database):
        """Test checking collection existence."""
        assert not database.has_collection("test")
        database.create_collection("test", dimension=64)
        assert database.has_collection("test")

    def test_list_collections(self, database):
        """Test listing all collections."""
        names = ["alpha", "beta", "gamma"]
        for name in names:
            database.create_collection(name, dimension=64)
        
        collections = database.list_collections()
        assert len(collections) == 3
        assert set(collections) == set(names)

    def test_delete_collection(self, database):
        """Test deleting a collection."""
        database.create_collection("test", dimension=64)
        assert database.has_collection("test")
        
        database.delete_collection("test")
        assert not database.has_collection("test")

    def test_delete_nonexistent_collection(self, database):
        """Test deleting a collection that doesn't exist."""
        with pytest.raises(CollectionNotFoundError):
            database.delete_collection("nonexistent")

    def test_get_or_create_collection(self, database):
        """Test get_or_create functionality."""
        # First call creates
        coll1 = database.get_or_create_collection("test", dimension=64)
        assert coll1 is not None
        
        # Second call gets existing
        coll2 = database.get_or_create_collection("test", dimension=64)
        assert coll2 is not None
        assert coll1.name == coll2.name


class TestVectorOperations:
    """Test vector CRUD operations across different scenarios."""

    def test_add_single_vector(self, database):
        """Test adding a single vector."""
        collection = database.create_collection("test", dimension=128)
        vector = np.random.randn(128).astype(np.float32)
        
        vector_id = collection.add(vector)
        assert vector_id is not None

    def test_add_vectors_with_ids(self, database):
        """Test adding vectors with custom IDs."""
        collection = database.create_collection("test", dimension=64)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"vec_{i}" for i in range(10)]
        
        returned_ids = collection.add(vectors, ids=ids)
        assert returned_ids == ids

    def test_add_vectors_with_metadata(self, database, sample_vectors, sample_metadata):
        """Test adding vectors with metadata."""
        collection = database.create_collection("test", dimension=128)
        
        ids = collection.add(
            sample_vectors[:100],
            metadata=sample_metadata[:100]
        )
        assert len(ids) == 100
        
        # Verify metadata is stored
        result = collection.get(ids[0])
        assert result["metadata"]["category"] in ["tech", "science", "art", "sports", "music"]

    def test_add_vectors_dimension_mismatch(self, database):
        """Test that dimension mismatch raises error."""
        collection = database.create_collection("test", dimension=64)
        wrong_vector = np.random.randn(128).astype(np.float32)
        
        with pytest.raises(DimensionMismatchError):
            collection.add(wrong_vector)

    def test_get_vector(self, database):
        """Test retrieving a vector by ID."""
        collection = database.create_collection("test", dimension=64)
        vector = np.random.randn(64).astype(np.float32)
        
        vector_id = collection.add(vector, ids=["test_id"])
        result = collection.get("test_id")
        
        assert result is not None
        assert np.allclose(result["vector"], vector, atol=1e-6)

    def test_get_nonexistent_vector(self, database):
        """Test getting a vector that doesn't exist."""
        collection = database.create_collection("test", dimension=64)
        
        with pytest.raises(VectorNotFoundError):
            collection.get("nonexistent")

    def test_get_multiple_vectors(self, database):
        """Test retrieving multiple vectors."""
        collection = database.create_collection("test", dimension=64)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"vec_{i}" for i in range(10)]
        
        collection.add(vectors, ids=ids)
        results = collection.get(ids[:5])
        
        assert len(results) == 5

    def test_update_vector(self, database):
        """Test updating a vector."""
        collection = database.create_collection("test", dimension=64)
        original = np.random.randn(64).astype(np.float32)
        updated = np.random.randn(64).astype(np.float32)
        
        collection.add(original, ids=["test_id"])
        collection.update("test_id", vector=updated)
        
        result = collection.get("test_id")
        assert np.allclose(result["vector"], updated, atol=1e-6)

    def test_update_metadata(self, database):
        """Test updating vector metadata."""
        collection = database.create_collection("test", dimension=64)
        vector = np.random.randn(64).astype(np.float32)
        
        collection.add(vector, ids=["test_id"], metadata=[{"status": "active"}])
        collection.update("test_id", metadata={"status": "inactive", "updated": True})
        
        result = collection.get("test_id")
        assert result["metadata"]["status"] == "inactive"
        assert result["metadata"]["updated"] is True

    def test_delete_vector(self, database):
        """Test deleting a vector."""
        collection = database.create_collection("test", dimension=64)
        vector = np.random.randn(64).astype(np.float32)
        
        collection.add(vector, ids=["test_id"])
        assert collection.count() == 1
        
        collection.delete("test_id")
        assert collection.count() == 0

    def test_delete_multiple_vectors(self, database):
        """Test deleting multiple vectors."""
        collection = database.create_collection("test", dimension=64)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"vec_{i}" for i in range(10)]
        
        collection.add(vectors, ids=ids)
        assert collection.count() == 10
        
        collection.delete(ids[:5])
        assert collection.count() == 5

    def test_collection_count(self, database, sample_vectors):
        """Test collection count accuracy."""
        collection = database.create_collection("test", dimension=128)
        
        assert collection.count() == 0
        
        collection.add(sample_vectors[:100])
        assert collection.count() == 100
        
        collection.add(sample_vectors[100:200])
        assert collection.count() == 200


class TestSearchOperations:
    """Test search functionality across index types."""

    @pytest.fixture
    def populated_collection(self, database, sample_vectors, sample_metadata):
        """Create a collection with sample data."""
        collection = database.create_collection("test", dimension=128)
        collection.add(sample_vectors, metadata=sample_metadata)
        return collection

    def test_basic_search(self, populated_collection):
        """Test basic similarity search."""
        query = np.random.randn(128).astype(np.float32)
        results = populated_collection.search(query, k=10)
        
        assert len(results) == 10
        assert all("id" in r for r in results)
        assert all("distance" in r for r in results)

    def test_search_returns_sorted_results(self, populated_collection):
        """Test that search results are sorted by distance."""
        query = np.random.randn(128).astype(np.float32)
        results = populated_collection.search(query, k=20)
        
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_search_with_different_k(self, populated_collection):
        """Test search with different k values."""
        query = np.random.randn(128).astype(np.float32)
        
        for k in [1, 5, 10, 50, 100]:
            results = populated_collection.search(query, k=k)
            assert len(results) == k

    def test_search_k_larger_than_collection(self, database):
        """Test search when k is larger than collection size."""
        collection = database.create_collection("small", dimension=64)
        vectors = np.random.randn(5, 64).astype(np.float32)
        collection.add(vectors)
        
        query = np.random.randn(64).astype(np.float32)
        results = collection.search(query, k=100)
        
        assert len(results) == 5

    def test_search_with_metadata_filter(self, populated_collection):
        """Test search with metadata filtering."""
        query = np.random.randn(128).astype(np.float32)
        results = populated_collection.search(
            query,
            k=50,
            filter={"category": "tech"}
        )
        
        assert len(results) <= 50
        assert all(r["metadata"]["category"] == "tech" for r in results)

    def test_search_with_range_filter(self, populated_collection):
        """Test search with range-based metadata filter."""
        query = np.random.randn(128).astype(np.float32)
        results = populated_collection.search(
            query,
            k=50,
            filter={"score": {"$gte": 0.5}}
        )
        
        assert all(r["metadata"]["score"] >= 0.5 for r in results)

    def test_search_with_boolean_filter(self, populated_collection):
        """Test search with boolean metadata filter."""
        query = np.random.randn(128).astype(np.float32)
        results = populated_collection.search(
            query,
            k=50,
            filter={"active": True}
        )
        
        assert all(r["metadata"]["active"] is True for r in results)

    def test_search_with_compound_filter(self, populated_collection):
        """Test search with compound filter conditions."""
        query = np.random.randn(128).astype(np.float32)
        results = populated_collection.search(
            query,
            k=50,
            filter={
                "$and": [
                    {"category": "tech"},
                    {"active": True},
                    {"score": {"$gte": 0.0}}
                ]
            }
        )
        
        for r in results:
            assert r["metadata"]["category"] == "tech"
            assert r["metadata"]["active"] is True

    def test_batch_search(self, populated_collection):
        """Test batch search with multiple queries."""
        queries = np.random.randn(5, 128).astype(np.float32)
        results = populated_collection.search_batch(queries, k=10)
        
        assert len(results) == 5
        assert all(len(r) == 10 for r in results)

    def test_search_with_include_vectors(self, populated_collection):
        """Test search with vector inclusion."""
        query = np.random.randn(128).astype(np.float32)
        results = populated_collection.search(query, k=5, include_vectors=True)
        
        assert all("vector" in r for r in results)
        assert all(len(r["vector"]) == 128 for r in results)


class TestIndexOperations:
    """Test index-specific operations."""

    def test_flat_index_exact_results(self, database):
        """Test that flat index returns exact results."""
        collection = database.create_collection(
            "test", dimension=64, index_type="flat"
        )
        vectors = np.random.randn(100, 64).astype(np.float32)
        ids = [f"vec_{i}" for i in range(100)]
        collection.add(vectors, ids=ids)
        
        # Search for exact vector
        results = collection.search(vectors[0], k=1)
        assert results[0]["id"] == "vec_0"

    def test_ivf_index_requires_training(self, database):
        """Test IVF index training behavior."""
        collection = database.create_collection(
            "test",
            dimension=64,
            index_type="ivf",
            index_params={"n_clusters": 10}
        )
        
        # Add enough vectors for training
        vectors = np.random.randn(500, 64).astype(np.float32)
        collection.add(vectors)
        
        # Should be able to search after auto-training
        query = np.random.randn(64).astype(np.float32)
        results = collection.search(query, k=10)
        assert len(results) == 10

    def test_hnsw_index_connectivity(self, database):
        """Test HNSW index maintains graph connectivity."""
        collection = database.create_collection(
            "test",
            dimension=64,
            index_type="hnsw",
            index_params={"M": 16, "ef_construction": 100}
        )
        
        vectors = np.random.randn(200, 64).astype(np.float32)
        collection.add(vectors)
        
        # Search should find results for any query
        for _ in range(10):
            query = np.random.randn(64).astype(np.float32)
            results = collection.search(query, k=10)
            assert len(results) == 10

    def test_index_rebuild(self, database):
        """Test index rebuild functionality."""
        collection = database.create_collection("test", dimension=64)
        vectors = np.random.randn(100, 64).astype(np.float32)
        collection.add(vectors)
        
        # Rebuild index
        collection.rebuild_index()
        
        # Search should still work
        query = np.random.randn(64).astype(np.float32)
        results = collection.search(query, k=10)
        assert len(results) == 10


class TestConcurrency:
    """Test concurrent database operations."""

    def test_concurrent_reads(self, database, sample_vectors):
        """Test concurrent read operations."""
        collection = database.create_collection("test", dimension=128)
        collection.add(sample_vectors[:100])
        
        def search_task(query_idx):
            query = sample_vectors[query_idx]
            results = collection.search(query, k=10)
            return len(results) == 10
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(search_task, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]
        
        assert all(results)

    def test_concurrent_writes(self, database):
        """Test concurrent write operations."""
        collection = database.create_collection("test", dimension=64)
        
        def add_task(batch_idx):
            vectors = np.random.randn(10, 64).astype(np.float32)
            ids = [f"batch_{batch_idx}_vec_{i}" for i in range(10)]
            collection.add(vectors, ids=ids)
            return True
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_task, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]
        
        assert all(results)
        assert collection.count() == 200

    def test_concurrent_read_write(self, database, sample_vectors):
        """Test concurrent read and write operations."""
        collection = database.create_collection("test", dimension=128)
        collection.add(sample_vectors[:100])
        
        errors = []
        
        def read_task():
            try:
                query = np.random.randn(128).astype(np.float32)
                collection.search(query, k=5)
            except Exception as e:
                errors.append(e)
        
        def write_task():
            try:
                vector = np.random.randn(128).astype(np.float32)
                collection.add(vector)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(20):
            threads.append(threading.Thread(target=read_task))
            threads.append(threading.Thread(target=write_task))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestDatabaseStats:
    """Test database statistics and monitoring."""

    def test_collection_stats(self, database, sample_vectors, sample_metadata):
        """Test collection statistics."""
        collection = database.create_collection("test", dimension=128)
        collection.add(sample_vectors[:100], metadata=sample_metadata[:100])
        
        stats = collection.stats()
        
        assert stats["count"] == 100
        assert stats["dimension"] == 128
        assert "index_type" in stats
        assert "memory_usage" in stats

    def test_database_stats(self, database):
        """Test database-level statistics."""
        database.create_collection("col1", dimension=64)
        database.create_collection("col2", dimension=128)
        
        stats = database.stats()
        
        assert stats["num_collections"] == 2
        assert "total_vectors" in stats
        assert "storage_size" in stats

    def test_collection_info(self, database):
        """Test collection information retrieval."""
        collection = database.create_collection(
            "test",
            dimension=128,
            metric="cosine",
            index_type="hnsw",
            description="Test collection"
        )
        
        info = collection.info()
        
        assert info["name"] == "test"
        assert info["dimension"] == 128
        assert info["metric"] == "cosine"
        assert info["index_type"] == "hnsw"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])