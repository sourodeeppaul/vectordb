"""
Unit tests for Collection class.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from vectordb.core.collection import (
    Collection,
    CollectionConfig,
    SearchResult,
    MetadataIndex,
)
from vectordb.core.exceptions import (
    VectorNotFoundError,
    VectorExistsError,
    DimensionMismatchError,
    ValidationError,
)


class TestCollectionBasics:
    """Basic collection tests."""
    
    def test_create_collection(self):
        """Test creating a collection."""
        collection = Collection("test", dimension=128)
        
        assert collection.name == "test"
        assert collection.dimension == 128
        assert collection.metric == "euclidean"
        assert len(collection) == 0
    
    def test_create_with_config(self):
        """Test creating with custom config."""
        collection = Collection(
            name="test",
            dimension=64,
            metric="cosine",
            normalize=True,
            max_vectors=1000,
        )
        
        assert collection.dimension == 64
        assert collection.metric == "cosine"
        assert collection.config.normalize_vectors is True
        assert collection.config.max_vectors == 1000
    
    def test_invalid_dimension(self):
        """Test rejection of invalid dimension."""
        with pytest.raises(ValidationError):
            Collection("test", dimension=0)
    
    def test_invalid_metric(self):
        """Test rejection of invalid metric."""
        with pytest.raises(ValidationError):
            Collection("test", dimension=128, metric="unknown")


class TestCollectionCRUD:
    """CRUD operation tests."""
    
    @pytest.fixture
    def collection(self):
        """Create a test collection."""
        return Collection("test", dimension=10)
    
    @pytest.fixture
    def sample_vector(self):
        """Create a sample vector."""
        return np.random.randn(10).astype(np.float32)
    
    # --- ADD ---
    
    def test_add_vector(self, collection, sample_vector):
        """Test adding a vector."""
        id = collection.add("vec1", sample_vector, {"key": "value"})
        
        assert id == "vec1"
        assert len(collection) == 1
        assert "vec1" in collection
    
    def test_add_generates_id(self, collection, sample_vector):
        """Test auto-generated ID."""
        id = collection.add("", sample_vector)
        
        assert id is not None
        assert len(id) > 0
    
    def test_add_duplicate_raises(self, collection, sample_vector):
        """Test duplicate ID rejection."""
        collection.add("vec1", sample_vector)
        
        with pytest.raises(VectorExistsError):
            collection.add("vec1", sample_vector)
    
    def test_add_overwrite(self, collection, sample_vector):
        """Test overwrite mode."""
        collection.add("vec1", sample_vector, {"v": 1})
        collection.add("vec1", sample_vector, {"v": 2}, overwrite=True)
        
        result = collection.get("vec1")
        assert result["metadata"]["v"] == 2
    
    def test_add_wrong_dimension(self, collection):
        """Test dimension mismatch rejection."""
        wrong_vector = np.random.randn(20).astype(np.float32)
        
        with pytest.raises(DimensionMismatchError):
            collection.add("vec1", wrong_vector)
    
    def test_add_from_list(self, collection):
        """Test adding from Python list."""
        vector_list = [0.1] * 10
        collection.add("vec1", vector_list)
        
        assert "vec1" in collection
    
    def test_add_batch(self, collection):
        """Test batch add."""
        vectors = [
            {"id": f"vec{i}", "vector": np.random.randn(10), "metadata": {"i": i}}
            for i in range(100)
        ]
        
        result = collection.add_batch(vectors)
        
        assert result["success_count"] == 100
        assert result["error_count"] == 0
        assert len(collection) == 100
    
    def test_add_normalized(self):
        """Test normalization on add."""
        collection = Collection("test", dimension=10, normalize=True)
        vector = np.array([3.0, 4.0] + [0.0] * 8, dtype=np.float32)
        
        collection.add("vec1", vector)
        result = collection.get("vec1")
        
        norm = np.linalg.norm(result["vector"])
        assert np.isclose(norm, 1.0)
    
    # --- GET ---
    
    def test_get_vector(self, collection, sample_vector):
        """Test getting a vector."""
        collection.add("vec1", sample_vector, {"key": "value"})
        
        result = collection.get("vec1")
        
        assert result["id"] == "vec1"
        assert np.allclose(result["vector"], sample_vector)
        assert result["metadata"]["key"] == "value"
    
    def test_get_not_found(self, collection):
        """Test getting non-existent vector."""
        result = collection.get("nonexistent")
        assert result is None
    
    def test_get_without_vector(self, collection, sample_vector):
        """Test getting without vector data."""
        collection.add("vec1", sample_vector)
        
        result = collection.get("vec1", include_vector=False)
        
        assert "vector" not in result
    
    def test_get_many(self, collection):
        """Test getting multiple vectors."""
        for i in range(5):
            collection.add(f"vec{i}", np.random.randn(10))
        
        results = collection.get_many(["vec0", "vec2", "vec4", "nonexistent"])
        
        assert len(results) == 4
        assert results[0]["id"] == "vec0"
        assert results[3] is None
    
    # --- UPDATE ---
    
    def test_update_vector(self, collection, sample_vector):
        """Test updating a vector."""
        collection.add("vec1", sample_vector)
        
        new_vector = np.random.randn(10).astype(np.float32)
        collection.update("vec1", vector=new_vector)
        
        result = collection.get("vec1")
        assert np.allclose(result["vector"], new_vector)
    
    def test_update_metadata(self, collection, sample_vector):
        """Test updating metadata."""
        collection.add("vec1", sample_vector, {"key": "old"})
        
        collection.update("vec1", metadata={"key": "new"})
        
        result = collection.get("vec1")
        assert result["metadata"]["key"] == "new"
    
    def test_update_metadata_merge(self, collection, sample_vector):
        """Test metadata merge mode."""
        collection.add("vec1", sample_vector, {"a": 1, "b": 2})
        
        collection.update(
            "vec1", 
            metadata={"b": 3, "c": 4},
            metadata_update_mode="merge"
        )
        
        result = collection.get("vec1")
        assert result["metadata"] == {"a": 1, "b": 3, "c": 4}
    
    def test_update_not_found(self, collection, sample_vector):
        """Test updating non-existent vector."""
        with pytest.raises(VectorNotFoundError):
            collection.update("nonexistent", vector=sample_vector)
    
    # --- DELETE ---
    
    def test_delete_vector(self, collection, sample_vector):
        """Test deleting a vector."""
        collection.add("vec1", sample_vector)
        
        result = collection.delete("vec1")
        
        assert result is True
        assert "vec1" not in collection
        assert len(collection) == 0
    
    def test_delete_not_found(self, collection):
        """Test deleting non-existent vector."""
        result = collection.delete("nonexistent")
        assert result is False
    
    def test_delete_many(self, collection):
        """Test deleting multiple vectors."""
        for i in range(5):
            collection.add(f"vec{i}", np.random.randn(10))
        
        result = collection.delete_many(["vec0", "vec2", "nonexistent"])
        
        assert result["deleted"] == 2
        assert result["not_found"] == 1
        assert len(collection) == 3
    
    def test_clear(self, collection):
        """Test clearing collection."""
        for i in range(10):
            collection.add(f"vec{i}", np.random.randn(10))
        
        count = collection.clear()
        
        assert count == 10
        assert len(collection) == 0


class TestCollectionSearch:
    """Search operation tests."""
    
    @pytest.fixture
    def populated_collection(self):
        """Create a populated collection."""
        collection = Collection("test", dimension=10, metric="euclidean")
        
        # Add vectors with metadata
        categories = ["A", "B", "C"]
        for i in range(100):
            vector = np.random.randn(10).astype(np.float32)
            metadata = {
                "category": categories[i % 3],
                "index": i,
                "value": float(i * 0.1),
            }
            collection.add(f"vec{i}", vector, metadata)
        
        return collection
    
    def test_basic_search(self, populated_collection):
        """Test basic similarity search."""
        query = np.random.randn(10).astype(np.float32)
        
        results = populated_collection.search(query, k=10)
        
        assert len(results) == 10
        assert all(isinstance(r, SearchResult) for r in results)
        # Results should be sorted by distance
        distances = [r.distance for r in results]
        assert distances == sorted(distances)
    
    def test_search_k_larger_than_collection(self):
        """Test search with k > collection size."""
        collection = Collection("test", dimension=10)
        for i in range(5):
            collection.add(f"vec{i}", np.random.randn(10))
        
        results = collection.search(np.random.randn(10), k=100)
        
        assert len(results) == 5
    
    def test_search_empty_collection(self):
        """Test search on empty collection."""
        collection = Collection("test", dimension=10)
        
        results = collection.search(np.random.randn(10), k=10)
        
        assert len(results) == 0
    
    def test_search_with_filter(self, populated_collection):
        """Test search with metadata filter."""
        query = np.random.randn(10).astype(np.float32)
        
        results = populated_collection.search(
            query, 
            k=10, 
            filter={"category": "A"}
        )
        
        assert len(results) <= 10
        assert all(r.metadata["category"] == "A" for r in results)
    
    def test_search_include_vector(self, populated_collection):
        """Test search with vector included."""
        query = np.random.randn(10).astype(np.float32)
        
        results = populated_collection.search(
            query, 
            k=5, 
            include_vector=True
        )
        
        assert all(r.vector is not None for r in results)
        assert all(len(r.vector) == 10 for r in results)
    
    def test_search_result_score(self, populated_collection):
        """Test that scores are computed correctly."""
        query = np.random.randn(10).astype(np.float32)
        
        results = populated_collection.search(query, k=10)
        
        # Scores should be in [0, 1] range
        for r in results:
            assert 0 <= r.score <= 1
        
        # First result should have highest score
        scores = [r.score for r in results]
        assert scores[0] >= scores[-1]
    
    def test_search_cosine_metric(self):
        """Test search with cosine metric."""
        collection = Collection("test", dimension=10, metric="cosine")
        
        # Add normalized vectors
        for i in range(50):
            v = np.random.randn(10).astype(np.float32)
            v /= np.linalg.norm(v)
            collection.add(f"vec{i}", v)
        
        query = np.random.randn(10).astype(np.float32)
        query /= np.linalg.norm(query)
        
        results = collection.search(query, k=5)
        
        assert len(results) == 5
        # Distances should be in [0, 2] for cosine
        assert all(0 <= r.distance <= 2 for r in results)
    
    def test_search_batch(self, populated_collection):
        """Test batch search."""
        queries = np.random.randn(5, 10).astype(np.float32)
        
        results = populated_collection.search_batch(queries, k=3)
        
        assert len(results) == 5
        assert all(len(r) == 3 for r in results)


class TestMetadataIndex:
    """Metadata index tests."""
    
    @pytest.fixture
    def index(self):
        return MetadataIndex()
    
    def test_add_and_query(self, index):
        """Test adding and querying."""
        index.add("id1", {"category": "A", "value": 1})
        index.add("id2", {"category": "A", "value": 2})
        index.add("id3", {"category": "B", "value": 1})
        
        results = index.query({"category": "A"})
        
        assert results == {"id1", "id2"}
    
    def test_query_multiple_fields(self, index):
        """Test AND logic for multiple fields."""
        index.add("id1", {"category": "A", "value": 1})
        index.add("id2", {"category": "A", "value": 2})
        index.add("id3", {"category": "B", "value": 1})
        
        results = index.query({"category": "A", "value": 1})
        
        assert results == {"id1"}
    
    def test_query_in_operator(self, index):
        """Test $in operator."""
        index.add("id1", {"category": "A"})
        index.add("id2", {"category": "B"})
        index.add("id3", {"category": "C"})
        
        results = index.query({"category": {"$in": ["A", "B"]}})
        
        assert results == {"id1", "id2"}
    
    def test_query_comparison_operators(self, index):
        """Test comparison operators."""
        index.add("id1", {"value": 1})
        index.add("id2", {"value": 2})
        index.add("id3", {"value": 3})
        
        # Greater than
        assert index.query({"value": {"$gt": 2}}) == {"id3"}
        
        # Less than or equal
        assert index.query({"value": {"$lte": 2}}) == {"id1", "id2"}
    
    def test_remove(self, index):
        """Test removing from index."""
        index.add("id1", {"category": "A"})
        index.add("id2", {"category": "A"})
        
        index.remove("id1", {"category": "A"})
        
        results = index.query({"category": "A"})
        assert results == {"id2"}
    
    def test_update(self, index):
        """Test updating index."""
        index.add("id1", {"category": "A"})
        
        index.update("id1", {"category": "A"}, {"category": "B"})
        
        assert index.query({"category": "A"}) == set()
        assert index.query({"category": "B"}) == {"id1"}


class TestCollectionSerialization:
    """Serialization tests."""
    
    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        collection = Collection("test", dimension=10, metric="cosine")
        
        # Add some data
        for i in range(10):
            collection.add(
                f"vec{i}",
                np.random.randn(10).astype(np.float32),
                {"index": i},
            )
        
        # Serialize and deserialize
        data = collection.to_dict()
        restored = Collection.from_dict(data)
        
        assert restored.name == collection.name
        assert restored.dimension == collection.dimension
        assert restored.metric == collection.metric
        assert len(restored) == len(collection)
        
        # Check a specific vector
        orig = collection.get("vec5")
        rest = restored.get("vec5")
        assert np.allclose(orig["vector"], rest["vector"])
        assert orig["metadata"] == rest["metadata"]
    
    def test_stats(self):
        """Test collection statistics."""
        collection = Collection("test", dimension=10)
        
        for i in range(100):
            collection.add(
                f"vec{i}",
                np.random.randn(10),
                {"data": "x" * 100},
            )
        
        stats = collection.stats()
        
        assert stats.name == "test"
        assert stats.dimension == 10
        assert stats.vector_count == 100
        assert stats.memory_usage_bytes > 0


class TestCollectionIteration:
    """Iteration tests."""
    
    def test_iter(self):
        """Test iterating over collection."""
        collection = Collection("test", dimension=10)
        
        for i in range(10):
            collection.add(f"vec{i}", np.random.randn(10))
        
        count = 0
        for record in collection:
            count += 1
            assert record.id.startswith("vec")
        
        assert count == 10
    
    def test_list_ids(self):
        """Test listing IDs with pagination."""
        collection = Collection("test", dimension=10)
        
        for i in range(100):
            collection.add(f"vec{i:03d}", np.random.randn(10))
        
        # Get first page
        page1 = collection.list_ids(limit=20, offset=0)
        assert len(page1) == 20
        
        # Get second page
        page2 = collection.list_ids(limit=20, offset=20)
        assert len(page2) == 20
        assert page1 != page2