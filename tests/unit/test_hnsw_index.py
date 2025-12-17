"""
Unit tests for HNSWIndex.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import time

from vectordb.index import HNSWIndex, SearchResult
from vectordb.index.hnsw import HNSWNode


class TestHNSWIndexBasics:
    """Basic HNSW tests."""
    
    @pytest.fixture
    def dimension(self):
        return 32
    
    @pytest.fixture
    def index(self, dimension):
        return HNSWIndex(dimension=dimension, metric="euclidean", M=8, seed=42)
    
    @pytest.fixture
    def random_vector(self, dimension):
        return np.random.randn(dimension).astype(np.float32)
    
    def test_create_index(self, dimension):
        """Test creating an index."""
        index = HNSWIndex(dimension=dimension, M=16)
        
        assert index.dimension == dimension
        assert index.metric == "euclidean"
        assert index.size == 0
        assert index.config.M == 16
    
    def test_create_with_options(self, dimension):
        """Test creating with custom options."""
        index = HNSWIndex(
            dimension=dimension,
            metric="cosine",
            M=32,
            M_max0=64,
            ef_construction=400,
            ef_search=100,
            normalize=True,
        )
        
        assert index.metric == "cosine"
        assert index.config.M == 32
        assert index.config.M_max0 == 64
        assert index.config.ef_construction == 400
        assert index.config.ef_search == 100
    
    def test_add_first_vector(self, index, random_vector):
        """Test adding first vector (entry point)."""
        index.add("vec1", random_vector, {"key": "value"})
        
        assert index.size == 1
        assert "vec1" in index
        assert index.entry_point == "vec1"
    
    def test_add_multiple_vectors(self, index, dimension):
        """Test adding multiple vectors."""
        for i in range(100):
            index.add(f"vec{i}", np.random.randn(dimension).astype(np.float32))
        
        assert index.size == 100
        assert index.max_level >= 0
    
    def test_add_duplicate_raises(self, index, random_vector):
        """Test duplicate ID rejection."""
        index.add("vec1", random_vector)
        
        with pytest.raises(ValueError, match="already exists"):
            index.add("vec1", random_vector)
    
    def test_add_wrong_dimension(self, index):
        """Test wrong dimension rejection."""
        wrong = np.random.randn(64).astype(np.float32)
        
        with pytest.raises(ValueError, match="dimension"):
            index.add("vec1", wrong)


class TestHNSWIndexSearch:
    """HNSW search tests."""
    
    @pytest.fixture
    def populated_index(self):
        """Create populated index."""
        np.random.seed(42)
        index = HNSWIndex(dimension=32, M=16, ef_construction=100, seed=42)
        
        for i in range(1000):
            vector = np.random.randn(32).astype(np.float32)
            metadata = {"index": i, "category": ["A", "B", "C"][i % 3]}
            index.add(f"vec{i}", vector, metadata)
        
        return index
    
    def test_basic_search(self, populated_index):
        """Test basic search."""
        query = np.random.randn(32).astype(np.float32)
        
        results = populated_index.search(query, k=10)
        
        assert len(results) == 10
        assert all(isinstance(r, SearchResult) for r in results)
        
        # Results should be sorted by distance
        distances = [r.distance for r in results]
        assert distances == sorted(distances)
    
    def test_search_recall(self, populated_index):
        """Test search recall against brute force."""
        from vectordb.index import FlatIndex
        
        # Build flat index for ground truth
        flat = FlatIndex(dimension=32, metric="euclidean")
        for id, vector, metadata in populated_index.iter_vectors():
            flat.add(id, vector, metadata)
        
        # Run queries
        np.random.seed(123)
        recalls = []
        
        for _ in range(10):
            query = np.random.randn(32).astype(np.float32)
            
            # Get HNSW results
            hnsw_results = populated_index.search(query, k=10, ef=100)
            hnsw_ids = {r.id for r in hnsw_results}
            
            # Get ground truth
            flat_results = flat.search(query, k=10)
            flat_ids = {r.id for r in flat_results}
            
            # Compute recall
            recall = len(hnsw_ids & flat_ids) / len(flat_ids)
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        assert avg_recall >= 0.8, f"Recall too low: {avg_recall}"
    
    def test_search_with_filter(self, populated_index):
        """Test search with filter."""
        query = np.random.randn(32).astype(np.float32)
        
        def filter_fn(id, metadata):
            return metadata.get("category") == "A"
        
        results = populated_index.search(query, k=10, filter_fn=filter_fn)
        
        assert all(r.metadata["category"] == "A" for r in results)
    
    def test_search_include_vectors(self, populated_index):
        """Test search with vectors included."""
        query = np.random.randn(32).astype(np.float32)
        
        results = populated_index.search(query, k=5, include_vectors=True)
        
        assert all(r.vector is not None for r in results)
        assert all(len(r.vector) == 32 for r in results)
    
    def test_search_batch(self, populated_index):
        """Test batch search."""
        queries = np.random.randn(5, 32).astype(np.float32)
        
        results = populated_index.search_batch(queries, k=10)
        
        assert len(results) == 5
        assert all(len(r) == 10 for r in results)
    
    def test_set_ef_search(self, populated_index):
        """Test adjusting ef_search."""
        query = np.random.randn(32).astype(np.float32)
        
        # Lower ef = faster but potentially lower recall
        populated_index.set_ef_search(10)
        results_low = populated_index.search(query, k=5)
        
        # Higher ef = slower but better recall
        populated_index.set_ef_search(200)
        results_high = populated_index.search(query, k=5)
        
        assert len(results_low) == 5
        assert len(results_high) == 5
    
    def test_search_with_ef_override(self, populated_index):
        """Test overriding ef at query time."""
        query = np.random.randn(32).astype(np.float32)
        
        results = populated_index.search(query, k=10, ef=200)
        
        assert len(results) == 10


class TestHNSWIndexRemove:
    """HNSW remove tests."""
    
    @pytest.fixture
    def index(self):
        index = HNSWIndex(dimension=32, M=8, seed=42)
        for i in range(100):
            index.add(f"vec{i}", np.random.randn(32).astype(np.float32))
        return index
    
    def test_remove(self, index):
        """Test removing a vector."""
        assert "vec50" in index
        
        result = index.remove("vec50")
        
        assert result is True
        assert "vec50" not in index
        assert index.size == 99
    
    def test_remove_not_found(self, index):
        """Test removing non-existent vector."""
        result = index.remove("nonexistent")
        
        assert result is False
    
    def test_remove_entry_point(self, index):
        """Test removing entry point."""
        ep = index.entry_point
        
        index.remove(ep)
        
        assert index.entry_point != ep
        assert index.entry_point is not None
    
    def test_remove_all(self, index):
        """Test removing all vectors."""
        for i in range(100):
            index.remove(f"vec{i}")
        
        assert index.size == 0
        assert index.entry_point is None
    
    def test_search_after_remove(self, index):
        """Test that search still works after removal."""
        # Remove some vectors
        for i in range(0, 50, 2):
            index.remove(f"vec{i}")
        
        query = np.random.randn(32).astype(np.float32)
        results = index.search(query, k=5)
        
        assert len(results) == 5
        # None of the removed vectors should be in results
        for r in results:
            idx = int(r.id.replace("vec", ""))
            assert idx >= 50 or idx % 2 == 1


class TestHNSWIndexGetContains:
    """Get and contains tests."""
    
    @pytest.fixture
    def index(self):
        index = HNSWIndex(dimension=10, seed=42)
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
    
    def test_contains(self, index):
        """Test contains check."""
        assert index.contains("vec0")
        assert "vec0" in index
        assert not index.contains("nonexistent")


class TestHNSWIndexSerialization:
    """Serialization tests."""
    
    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        np.random.seed(42)
        
        index = HNSWIndex(dimension=16, metric="cosine", M=8, seed=42)
        
        for i in range(100):
            index.add(f"vec{i}", np.random.randn(16).astype(np.float32), {"i": i})
        
        # Serialize
        data = index.to_dict()
        
        # Deserialize
        restored = HNSWIndex.from_dict(data)
        
        assert restored.dimension == index.dimension
        assert restored.metric == index.metric
        assert restored.size == index.size
        assert restored.max_level == index.max_level
        assert restored.entry_point == index.entry_point
        
        # Check vectors match
        for id in ["vec0", "vec50", "vec99"]:
            orig_vec, orig_meta = index.get(id)
            rest_vec, rest_meta = restored.get(id)
            
            assert_array_almost_equal(orig_vec, rest_vec)
            assert orig_meta == rest_meta
        
        # Check search works
        query = np.random.randn(16).astype(np.float32)
        orig_results = index.search(query, k=5)
        rest_results = restored.search(query, k=5)
        
        # Results should be similar (not necessarily identical due to randomness)
        orig_ids = {r.id for r in orig_results}
        rest_ids = {r.id for r in rest_results}
        
        # At least 3 of 5 should overlap
        assert len(orig_ids & rest_ids) >= 3


class TestHNSWIndexStats:
    """Statistics tests."""
    
    def test_stats(self):
        """Test index statistics."""
        np.random.seed(42)
        index = HNSWIndex(dimension=32, M=16, seed=42)
        
        for i in range(1000):
            index.add(f"vec{i}", np.random.randn(32).astype(np.float32))
        
        stats = index.stats()
        
        assert stats.index_type == "hnsw"
        assert stats.dimension == 32
        assert stats.vector_count == 1000
        assert stats.memory_bytes > 0
        assert stats.extra["M"] == 16
        assert stats.extra["max_level"] >= 0
        assert stats.extra["total_connections"] > 0
    
    def test_graph_info(self):
        """Test graph info."""
        np.random.seed(42)
        index = HNSWIndex(dimension=16, M=8, seed=42)
        
        for i in range(500):
            index.add(f"vec{i}", np.random.randn(16).astype(np.float32))
        
        info = index.get_graph_info()
        
        assert "max_level" in info
        assert "entry_point" in info
        assert "levels" in info
        assert len(info["levels"]) > 0


class TestHNSWIndexGraphStructure:
    """Graph structure tests."""
    
    def test_level_distribution(self):
        """Test that level distribution follows expected pattern."""
        np.random.seed(42)
        index = HNSWIndex(dimension=16, M=16, seed=42)
        
        for i in range(1000):
            index.add(f"vec{i}", np.random.randn(16).astype(np.float32))
        
        # Count nodes at each level
        level_counts = {}
        for id in index.iter_ids():
            level = index.get_node_level(id)
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Most nodes should be at low levels
        # Higher levels should have exponentially fewer nodes
        assert level_counts.get(0, 0) > level_counts.get(1, 0)
    
    def test_connectivity(self):
        """Test that graph is connected."""
        np.random.seed(42)
        index = HNSWIndex(dimension=16, M=8, seed=42)
        
        for i in range(100):
            index.add(f"vec{i}", np.random.randn(16).astype(np.float32))
        
        # Every node should have neighbors at level 0
        for id in index.iter_ids():
            neighbors = index.get_neighbors(id, level=0)
            assert len(neighbors) > 0 or index.size == 1
    
    def test_neighbor_count_within_limits(self):
        """Test that neighbor counts are within M limits."""
        np.random.seed(42)
        index = HNSWIndex(dimension=16, M=8, seed=42)
        
        for i in range(200):
            index.add(f"vec{i}", np.random.randn(16).astype(np.float32))
        
        for id in index.iter_ids():
            node = index._nodes[id]
            
            # Check level 0
            assert len(node.get_neighbors(0)) <= index.config.M_max0
            
            # Check higher levels
            for level in range(1, node.layer + 1):
                assert len(node.get_neighbors(level)) <= index.config.M


class TestHNSWIndexNormalization:
    """Normalization tests."""
    
    def test_normalized_vectors(self):
        """Test that vectors are normalized when enabled."""
        index = HNSWIndex(dimension=10, normalize=True, seed=42)
        
        vector = np.array([3.0, 4.0] + [0.0] * 8, dtype=np.float32)
        index.add("vec1", vector)
        
        stored, _ = index.get("vec1")
        norm = np.linalg.norm(stored)
        
        assert np.isclose(norm, 1.0)


class TestHNSWIndexIteration:
    """Iteration tests."""
    
    @pytest.fixture
    def index(self):
        index = HNSWIndex(dimension=10, seed=42)
        for i in range(20):
            index.add(f"vec{i}", np.random.randn(10).astype(np.float32))
        return index
    
    def test_iter_ids(self, index):
        """Test iterating over IDs."""
        ids = list(index.iter_ids())
        
        assert len(ids) == 20
        assert "vec0" in ids
    
    def test_iter_vectors(self, index):
        """Test iterating over vectors."""
        count = 0
        for id, vector, metadata in index.iter_vectors():
            assert len(vector) == 10
            count += 1
        
        assert count == 20


class TestHNSWIndexRebuild:
    """Rebuild tests."""
    
    def test_rebuild(self):
        """Test rebuilding index."""
        np.random.seed(42)
        index = HNSWIndex(dimension=16, M=8, seed=42)
        
        for i in range(100):
            index.add(f"vec{i}", np.random.randn(16).astype(np.float32))
        
        # Remove some
        for i in range(0, 50, 2):
            index.remove(f"vec{i}")
        
        original_size = index.size
        
        # Rebuild
        index.rebuild()
        
        assert index.size == original_size
        
        # Search should still work
        query = np.random.randn(16).astype(np.float32)
        results = index.search(query, k=5)
        assert len(results) == 5


class TestHNSWIndexPerformance:
    """Performance tests (marked as slow)."""
    
    @pytest.mark.slow
    def test_large_index(self):
        """Test with larger dataset."""
        np.random.seed(42)
        index = HNSWIndex(dimension=128, M=16, ef_construction=100, seed=42)
        
        n_vectors = 10000
        vectors = np.random.randn(n_vectors, 128).astype(np.float32)
        
        # Build
        start = time.time()
        for i in range(n_vectors):
            index.add(f"vec{i}", vectors[i])
        build_time = time.time() - start
        
        print(f"\nBuild time: {build_time:.2f}s ({n_vectors/build_time:.0f} vec/s)")
        
        # Search
        n_queries = 100
        queries = np.random.randn(n_queries, 128).astype(np.float32)
        
        start = time.time()
        for q in queries:
            index.search(q, k=10)
        search_time = time.time() - start
        
        qps = n_queries / search_time
        print(f"Search: {qps:.0f} QPS, {search_time/n_queries*1000:.2f}ms per query")
        
        assert qps > 100  # At least 100 QPS
    
    @pytest.mark.slow
    def test_recall_vs_ef(self):
        """Test recall at different ef values."""
        from vectordb.index import FlatIndex
        
        np.random.seed(42)
        
        # Build indices
        dimension = 64
        n_vectors = 5000
        
        hnsw = HNSWIndex(dimension=dimension, M=16, ef_construction=200, seed=42)
        flat = FlatIndex(dimension=dimension, metric="euclidean")
        
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        
        for i in range(n_vectors):
            hnsw.add(f"vec{i}", vectors[i])
            flat.add(f"vec{i}", vectors[i])
        
        # Test different ef values
        ef_values = [10, 20, 50, 100, 200]
        k = 10
        n_queries = 50
        queries = np.random.randn(n_queries, dimension).astype(np.float32)
        
        print("\n\nef_search vs Recall:")
        print("-" * 40)
        
        for ef in ef_values:
            recalls = []
            
            for q in queries:
                hnsw_results = hnsw.search(q, k=k, ef=ef)
                flat_results = flat.search(q, k=k)
                
                hnsw_ids = {r.id for r in hnsw_results}
                flat_ids = {r.id for r in flat_results}
                
                recall = len(hnsw_ids & flat_ids) / k
                recalls.append(recall)
            
            avg_recall = np.mean(recalls)
            print(f"ef={ef:3d}: recall={avg_recall:.3f}")
            
            if ef >= 100:
                assert avg_recall >= 0.9