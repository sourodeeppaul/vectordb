"""
Unit tests for FlatIndex.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from vectordb.index import FlatIndex, SearchResult


class TestFlatIndexBasics:
    """Basic FlatIndex tests."""
    
    @pytest.fixture
    def dimension(self):
        return 128
    
    @pytest.fixture
    def index(self, dimension):
        return FlatIndex(dimension=dimension, metric="euclidean")
    
    @pytest.fixture
    def random_vector(self, dimension):
        return np.random.randn(dimension).astype(np.float32)
    
    def test_create_index(self, dimension):
        """Test creating an index."""
        index = FlatIndex(dimension=dimension)
        
        assert index.dimension == dimension
        assert index.metric == "euclidean"
        assert index.size == 0
        assert len(index) == 0
    
    def test_create_with_options(self, dimension):
        """Test creating with custom options."""
        index = FlatIndex(
            dimension=dimension,
            metric="cosine",
            normalize=True,
            use_matrix=True,
        )
        
        assert index.metric == "cosine"
        assert index.config.normalize is True
    
    def test_add_vector(self, index, random_vector):
        """Test adding a vector."""
        index.add("vec1", random_vector, {"key": "value"})
        
        assert index.size == 1
        assert "vec1" in index
    
    def test_add_duplicate_raises(self, index, random_vector):
        """Test duplicate ID rejection."""
        index.add("vec1", random_vector)
        
        with pytest.raises(ValueError, match="already exists"):
            index.add("vec1", random_vector)
    
    def test_add_wrong_dimension(self, index):
        """Test wrong dimension rejection."""
        wrong_vector = np.random.randn(64).astype(np.float32)
        
        with pytest.raises(ValueError, match="dimension"):
            index.add("vec1", wrong_vector)
    
    def test_add_from_list(self, index, dimension):
        """Test adding from Python list."""
        vector_list = [0.1] * dimension
        index.add("vec1", vector_list)
        
        assert "vec1" in index


class TestFlatIndexBatch:
    """Batch operation tests."""
    
    @pytest.fixture
    def index(self):
        return FlatIndex(dimension=10)
    
    def test_add_batch(self, index):
        """Test batch add."""
        ids = [f"vec{i}" for i in range(100)]
        vectors = np.random.randn(100, 10).astype(np.float32)
        metadata = [{"index": i} for i in range(100)]
        
        count = index.add_batch(ids, vectors, metadata)
        
        assert count == 100
        assert index.size == 100
    
    def test_add_batch_no_metadata(self, index):
        """Test batch add without metadata."""
        ids = [f"vec{i}" for i in range(50)]
        vectors = np.random.randn(50, 10).astype(np.float32)
        
        count = index.add_batch(ids, vectors)
        
        assert count == 50
    
    def test_add_batch_duplicate(self, index):
        """Test batch add with duplicate ID."""
        index.add("vec0", np.random.randn(10))
        
        ids = [f"vec{i}" for i in range(5)]
        vectors = np.random.randn(5, 10).astype(np.float32)
        
        with pytest.raises(ValueError, match="already exists"):
            index.add_batch(ids, vectors)
    
    def test_add_batch_mismatched_counts(self, index):
        """Test batch add with mismatched counts."""
        ids = ["vec0", "vec1"]
        vectors = np.random.randn(5, 10).astype(np.float32)
        
        with pytest.raises(ValueError):
            index.add_batch(ids, vectors)


class TestFlatIndexSearch:
    """Search operation tests."""
    
    @pytest.fixture
    def populated_index(self):
        """Create populated index."""
        index = FlatIndex(dimension=10, metric="euclidean")
        
        for i in range(100):
            vector = np.random.randn(10).astype(np.float32)
            metadata = {"index": i, "category": ["A", "B", "C"][i % 3]}
            index.add(f"vec{i}", vector, metadata)
        
        return index
    
    def test_basic_search(self, populated_index):
        """Test basic search."""
        query = np.random.randn(10).astype(np.float32)
        
        results = populated_index.search(query, k=10)
        
        assert len(results) == 10
        assert all(isinstance(r, SearchResult) for r in results)
        
        # Results should be sorted by distance
        distances = [r.distance for r in results]
        assert distances == sorted(distances)
    
    def test_search_k_larger_than_size(self, populated_index):
        """Test search with k > index size."""
        query = np.random.randn(10).astype(np.float32)
        
        results = populated_index.search(query, k=1000)
        
        assert len(results) == 100  # Capped at index size
    
    def test_search_empty_index(self):
        """Test search on empty index."""
        index = FlatIndex(dimension=10)
        query = np.random.randn(10).astype(np.float32)
        
        results = index.search(query, k=10)
        
        assert len(results) == 0
    
    def test_search_with_filter(self, populated_index):
        """Test search with filter function."""
        query = np.random.randn(10).astype(np.float32)
        
        def filter_fn(id, metadata):
            return metadata.get("category") == "A"
        
        results = populated_index.search(query, k=10, filter_fn=filter_fn)
        
        assert all(r.metadata["category"] == "A" for r in results)
    
    def test_search_include_vectors(self, populated_index):
        """Test search with vectors included."""
        query = np.random.randn(10).astype(np.float32)
        
        results = populated_index.search(query, k=5, include_vectors=True)
        
        assert all(r.vector is not None for r in results)
        assert all(len(r.vector) == 10 for r in results)
    
    def test_search_batch(self, populated_index):
        """Test batch search."""
        queries = np.random.randn(5, 10).astype(np.float32)
        
        results = populated_index.search_batch(queries, k=3)
        
        assert len(results) == 5
        assert all(len(r) == 3 for r in results)
    
    def test_search_by_id(self, populated_index):
        """Test search by existing vector ID."""
        results = populated_index.search_by_id("vec50", k=5)
        
        assert len(results) == 5
        assert all(r.id != "vec50" for r in results)  # Excluded self
    
    def test_search_cosine(self):
        """Test search with cosine metric."""
        index = FlatIndex(dimension=10, metric="cosine", normalize=True)
        
        for i in range(50):
            vector = np.random.randn(10).astype(np.float32)
            index.add(f"vec{i}", vector)
        
        query = np.random.randn(10).astype(np.float32)
        results = index.search(query, k=5)
        
        assert len(results) == 5
        # Cosine distances should be in [0, 2]
        assert all(0 <= r.distance <= 2 for r in results)


class TestFlatIndexRangeSearch:
    """Range search tests."""
    
    @pytest.fixture
    def index(self):
        """Create index with known vectors."""
        index = FlatIndex(dimension=2, metric="euclidean")
        
        # Add vectors at known positions
        positions = [
            (0, 0), (1, 0), (0, 1), (1, 1),
            (5, 5), (6, 5), (5, 6), (6, 6),
        ]
        
        for i, (x, y) in enumerate(positions):
            index.add(f"vec{i}", np.array([x, y], dtype=np.float32))
        
        return index
    
    def test_range_search(self, index):
        """Test range search."""
        query = np.array([0, 0], dtype=np.float32)
        
        results = index.range_search(query, radius=1.5)
        
        # Should find vectors at (0,0), (1,0), (0,1), (1,1)
        assert len(results) == 4
    
    def test_range_search_with_filter(self, index):
        """Test range search with filter."""
        query = np.array([0, 0], dtype=np.float32)
        
        def filter_fn(id, meta):
            return id in ["vec0", "vec1"]
        
        results = index.range_search(query, radius=1.5, filter_fn=filter_fn)
        
        assert len(results) == 2


class TestFlatIndexRemove:
    """Remove operation tests."""
    
    @pytest.fixture
    def populated_index(self):
        index = FlatIndex(dimension=10)
        for i in range(100):
            index.add(f"vec{i}", np.random.randn(10).astype(np.float32))
        return index
    
    def test_remove(self, populated_index):
        """Test removing a vector."""
        result = populated_index.remove("vec50")
        
        assert result is True
        assert "vec50" not in populated_index
        assert populated_index.size == 99
    
    def test_remove_not_found(self, populated_index):
        """Test removing non-existent vector."""
        result = populated_index.remove("nonexistent")
        
        assert result is False
    
    def test_remove_batch(self, populated_index):
        """Test batch remove."""
        count = populated_index.remove_batch(["vec0", "vec1", "vec2", "nonexistent"])
        
        assert count == 3
        assert populated_index.size == 97
    
    def test_clear(self, populated_index):
        """Test clearing index."""
        count = populated_index.clear()
        
        assert count == 100
        assert populated_index.size == 0


class TestFlatIndexGet:
    """Get operation tests."""
    
    @pytest.fixture
    def index(self):
        index = FlatIndex(dimension=10)
        for i in range(10):
            vector = np.arange(i, i + 10, dtype=np.float32)
            index.add(f"vec{i}", vector, {"index": i})
        return index
    
    def test_get(self, index):
        """Test getting a vector."""
        result = index.get("vec5")
        
        assert result is not None
        vector, metadata = result
        assert_array_almost_equal(vector, np.arange(5, 15, dtype=np.float32))
        assert metadata["index"] == 5
    
    def test_get_not_found(self, index):
        """Test getting non-existent vector."""
        result = index.get("nonexistent")
        
        assert result is None
    
    def test_get_vectors(self, index):
        """Test getting multiple vectors."""
        results = index.get_vectors(["vec0", "vec5", "nonexistent"])
        
        assert results["vec0"] is not None
        assert results["vec5"] is not None
        assert results["nonexistent"] is None
    
    def test_contains(self, index):
        """Test contains check."""
        assert index.contains("vec0")
        assert "vec0" in index
        assert not index.contains("nonexistent")


class TestFlatIndexUpdate:
    """Update operation tests."""
    
    @pytest.fixture
    def index(self):
        index = FlatIndex(dimension=10)
        index.add("vec1", np.zeros(10, dtype=np.float32), {"key": "old"})
        return index
    
    def test_update_vector(self, index):
        """Test updating vector."""
        new_vector = np.ones(10, dtype=np.float32)
        
        result = index.update("vec1", vector=new_vector)
        
        assert result is True
        vector, _ = index.get("vec1")
        assert_array_almost_equal(vector, new_vector)
    
    def test_update_metadata(self, index):
        """Test updating metadata."""
        result = index.update("vec1", metadata={"key": "new"})
        
        assert result is True
        _, metadata = index.get("vec1")
        assert metadata["key"] == "new"
    
    def test_update_not_found(self, index):
        """Test updating non-existent vector."""
        result = index.update("nonexistent", vector=np.ones(10))
        
        assert result is False


class TestFlatIndexSerialization:
    """Serialization tests."""
    
    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        index = FlatIndex(dimension=10, metric="cosine", normalize=True)
        
        for i in range(50):
            index.add(f"vec{i}", np.random.randn(10).astype(np.float32), {"i": i})
        
        # Serialize
        data = index.to_dict()
        
        # Deserialize
        restored = FlatIndex.from_dict(data)
        
        assert restored.dimension == index.dimension
        assert restored.metric == index.metric
        assert restored.size == index.size
        
        # Check vectors match
        for id in ["vec0", "vec25", "vec49"]:
            orig_vec, orig_meta = index.get(id)
            rest_vec, rest_meta = restored.get(id)
            
            assert_array_almost_equal(orig_vec, rest_vec)
            assert orig_meta == rest_meta


class TestFlatIndexStats:
    """Statistics tests."""
    
    def test_stats(self):
        """Test index statistics."""
        index = FlatIndex(dimension=128)
        
        for i in range(1000):
            index.add(f"vec{i}", np.random.randn(128).astype(np.float32))
        
        stats = index.stats()
        
        assert stats.index_type == "flat"
        assert stats.dimension == 128
        assert stats.vector_count == 1000
        assert stats.memory_bytes > 0
        assert stats.is_trained is True


class TestFlatIndexNormalization:
    """Normalization tests."""
    
    def test_normalized_vectors(self):
        """Test that vectors are normalized when enabled."""
        index = FlatIndex(dimension=10, normalize=True)
        
        # Add unnormalized vector
        vector = np.array([3.0, 4.0] + [0.0] * 8, dtype=np.float32)
        index.add("vec1", vector)
        
        # Retrieved vector should be normalized
        stored, _ = index.get("vec1")
        norm = np.linalg.norm(stored)
        
        assert np.isclose(norm, 1.0)


class TestFlatIndexDictStorage:
    """Tests for dict-based storage (use_matrix=False)."""
    
    def test_dict_storage(self):
        """Test using dict storage instead of matrix."""
        index = FlatIndex(dimension=10, use_matrix=False)
        
        for i in range(50):
            index.add(f"vec{i}", np.random.randn(10).astype(np.float32))
        
        assert index.size == 50
        
        # Search should still work
        results = index.search(np.random.randn(10).astype(np.float32), k=5)
        assert len(results) == 5
    
    def test_remove_dict_storage(self):
        """Test remove with dict storage."""
        index = FlatIndex(dimension=10, use_matrix=False)
        
        for i in range(10):
            index.add(f"vec{i}", np.random.randn(10).astype(np.float32))
        
        index.remove("vec5")
        
        assert index.size == 9
        assert "vec5" not in index


class TestFlatIndexDuplicates:
    """Duplicate detection tests."""
    
    def test_find_duplicates(self):
        """Test finding duplicate vectors."""
        index = FlatIndex(dimension=10)
        
        # Add some vectors with duplicates
        v1 = np.random.randn(10).astype(np.float32)
        index.add("vec1", v1)
        index.add("vec2", v1.copy())  # Duplicate
        index.add("vec3", np.random.randn(10).astype(np.float32))
        
        duplicates = index.find_duplicates(threshold=1e-5)
        
        assert len(duplicates) == 1
        assert ("vec1", "vec2") == (duplicates[0][0], duplicates[0][1]) or \
               ("vec2", "vec1") == (duplicates[0][0], duplicates[0][1])


class TestFlatIndexIteration:
    """Iteration tests."""
    
    @pytest.fixture
    def index(self):
        index = FlatIndex(dimension=10)
        for i in range(10):
            index.add(f"vec{i}", np.random.randn(10).astype(np.float32))
        return index
    
    def test_iter_ids(self, index):
        """Test iterating over IDs."""
        ids = list(index.iter_ids())
        
        assert len(ids) == 10
        assert "vec0" in ids
    
    def test_iter_vectors(self, index):
        """Test iterating over vectors."""
        count = 0
        for id, vector, metadata in index.iter_vectors():
            assert len(vector) == 10
            count += 1
        
        assert count == 10
    
    def test_get_all_vectors(self, index):
        """Test getting all vectors."""
        vectors = index.get_all_vectors()
        
        assert vectors.shape == (10, 10)
    
    def test_get_all_ids(self, index):
        """Test getting all IDs."""
        ids = index.get_all_ids()
        
        assert len(ids) == 10
    
    def test_sample(self, index):
        """Test random sampling."""
        samples = index.sample(3, seed=42)
        
        assert len(samples) == 3
        for id, vector, metadata in samples:
            assert id in index