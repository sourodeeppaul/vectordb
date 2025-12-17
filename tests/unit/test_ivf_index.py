"""
Unit tests for IVFIndex.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import time

from vectordb.index import IVFIndex, FlatIndex, SearchResult
from vectordb.index.ivf import KMeans


class TestKMeans:
    """K-Means tests."""
    
    def test_fit_basic(self):
        """Test basic k-means fitting."""
        np.random.seed(42)
        vectors = np.random.randn(100, 10).astype(np.float32)
        
        kmeans = KMeans(n_clusters=5, seed=42)
        centroids = kmeans.fit(vectors)
        
        assert centroids.shape == (5, 10)
    
    def test_predict(self):
        """Test cluster prediction."""
        np.random.seed(42)
        vectors = np.random.randn(100, 10).astype(np.float32)
        
        kmeans = KMeans(n_clusters=5, seed=42)
        kmeans.fit(vectors)
        
        assignments = kmeans.predict(vectors)
        
        assert len(assignments) == 100
        assert all(0 <= a < 5 for a in assignments)
    
    def test_few_samples(self):
        """Test with fewer samples than clusters."""
        np.random.seed(42)
        vectors = np.random.randn(3, 10).astype(np.float32)
        
        kmeans = KMeans(n_clusters=10, seed=42)
        centroids = kmeans.fit(vectors)
        
        # Should use all samples as centroids
        assert len(centroids) == 3


class TestIVFIndexBasics:
    """Basic IVF index tests."""
    
    @pytest.fixture
    def dimension(self):
        return 32
    
    @pytest.fixture
    def training_vectors(self, dimension):
        np.random.seed(42)
        return np.random.randn(1000, dimension).astype(np.float32)
    
    @pytest.fixture
    def trained_index(self, dimension, training_vectors):
        index = IVFIndex(dimension=dimension, n_clusters=10, n_probe=3, seed=42)
        index.train(training_vectors)
        return index
    
    def test_create_index(self, dimension):
        """Test creating an index."""
        index = IVFIndex(dimension=dimension, n_clusters=50, n_probe=5)
        
        assert index.dimension == dimension
        assert index.n_clusters == 50
        assert index.n_probe == 5
        assert not index.is_trained()
    
    def test_train(self, dimension, training_vectors):
        """Test training the index."""
        index = IVFIndex(dimension=dimension, n_clusters=20, seed=42)
        
        index.train(training_vectors)
        
        assert index.is_trained()
        assert index.centroids is not None
        assert index.centroids.shape == (20, dimension)
    
    def test_add_before_train_raises(self, dimension):
        """Test that add before train raises error."""
        index = IVFIndex(dimension=dimension, n_clusters=10)
        vector = np.random.randn(dimension).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="trained"):
            index.add("vec1", vector)
    
    def test_add_vector(self, trained_index, dimension):
        """Test adding a vector."""
        vector = np.random.randn(dimension).astype(np.float32)
        
        trained_index.add("vec1", vector, {"key": "value"})
        
        assert trained_index.size == 1
        assert "vec1" in trained_index
    
    def test_add_duplicate_raises(self, trained_index, dimension):
        """Test duplicate ID rejection."""
        vector = np.random.randn(dimension).astype(np.float32)
        
        trained_index.add("vec1", vector)
        
        with pytest.raises(ValueError, match="already exists"):
            trained_index.add("vec1", vector)
    
    def test_add_batch(self, trained_index, dimension):
        """Test batch add."""
        ids = [f"vec{i}" for i in range(100)]
        vectors = np.random.randn(100, dimension).astype(np.float32)
        
        count = trained_index.add_batch(ids, vectors)
        
        assert count == 100
        assert trained_index.size == 100


class TestIVFIndexSearch:
    """IVF search tests."""
    
    @pytest.fixture
    def populated_index(self):
        """Create populated index."""
        np.random.seed(42)
        dimension = 32
        n_vectors = 5000
        
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        
        index = IVFIndex(dimension=dimension, n_clusters=50, n_probe=5, seed=42)
        index.train(vectors[:1000])
        
        for i in range(n_vectors):
            metadata = {"index": i, "category": ["A", "B", "C"][i % 3]}
            index.add(f"vec{i}", vectors[i], metadata)
        
        return index, vectors
    
    def test_basic_search(self, populated_index):
        """Test basic search."""
        index, vectors = populated_index
        query = np.random.randn(32).astype(np.float32)
        
        results = index.search(query, k=10)
        
        assert len(results) == 10
        assert all(isinstance(r, SearchResult) for r in results)
        
        # Results should be sorted by distance
        distances = [r.distance for r in results]
        assert distances == sorted(distances)
    
    def test_search_recall(self, populated_index):
        """Test search recall against brute force."""
        index, vectors = populated_index
        
        # Build flat index for ground truth
        flat = FlatIndex(dimension=32, metric="euclidean")
        for i in range(len(vectors)):
            flat.add(f"vec{i}", vectors[i])
        
        # Run queries
        np.random.seed(123)
        recalls = []
        
        for _ in range(20):
            query = np.random.randn(32).astype(np.float32)
            
            # Get IVF results with high n_probe
            ivf_results = index.search(query, k=10, n_probe=20)
            ivf_ids = {r.id for r in ivf_results}
            
            # Get ground truth
            flat_results = flat.search(query, k=10)
            flat_ids = {r.id for r in flat_results}
            
            # Compute recall
            recall = len(ivf_ids & flat_ids) / len(flat_ids)
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        assert avg_recall >= 0.7, f"Recall too low: {avg_recall}"
    
    def test_search_with_filter(self, populated_index):
        """Test search with filter."""
        index, _ = populated_index
        query = np.random.randn(32).astype(np.float32)
        
        def filter_fn(id, metadata):
            return metadata.get("category") == "A"
        
        results = index.search(query, k=10, filter_fn=filter_fn)
        
        assert all(r.metadata["category"] == "A" for r in results)
    
    def test_search_include_vectors(self, populated_index):
        """Test search with vectors included."""
        index, _ = populated_index
        query = np.random.randn(32).astype(np.float32)
        
        results = index.search(query, k=5, include_vectors=True)
        
        assert all(r.vector is not None for r in results)
        assert all(len(r.vector) == 32 for r in results)
    
    def test_search_batch(self, populated_index):
        """Test batch search."""
        index, _ = populated_index
        queries = np.random.randn(5, 32).astype(np.float32)
        
        results = index.search_batch(queries, k=10)
        
        assert len(results) == 5
        assert all(len(r) == 10 for r in results)
    
    def test_set_n_probe(self, populated_index):
        """Test adjusting n_probe."""
        index, _ = populated_index
        query = np.random.randn(32).astype(np.float32)
        
        # Lower n_probe = faster but potentially lower recall
        index.set_n_probe(1)
        results_low = index.search(query, k=5)
        
        # Higher n_probe = slower but better recall
        index.set_n_probe(30)
        results_high = index.search(query, k=5)
        
        assert len(results_low) == 5
        assert len(results_high) == 5
    
    def test_search_with_n_probe_override(self, populated_index):
        """Test overriding n_probe at query time."""
        index, _ = populated_index
        query = np.random.randn(32).astype(np.float32)
        
        results = index.search(query, k=10, n_probe=25)
        
        assert len(results) == 10


class TestIVFIndexRecallVsNProbe:
    """Test recall at different n_probe values."""
    
    @pytest.mark.slow
    def test_recall_vs_n_probe(self):
        """Test that recall improves with higher n_probe."""
        np.random.seed(42)
        dimension = 64
        n_vectors = 10000
        
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        
        # Build IVF index
        ivf = IVFIndex(dimension=dimension, n_clusters=100, seed=42)
        ivf.train(vectors[:2000])
        
        for i in range(n_vectors):
            ivf.add(f"vec{i}", vectors[i])
        
        # Build flat index for ground truth
        flat = FlatIndex(dimension=dimension)
        for i in range(n_vectors):
            flat.add(f"vec{i}", vectors[i])
        
        # Test different n_probe values
        n_probe_values = [1, 5, 10, 20, 50]
        recalls = {}
        
        queries = np.random.randn(20, dimension).astype(np.float32)
        k = 10
        
        for n_probe in n_probe_values:
            query_recalls = []
            
            for q in queries:
                ivf_results = ivf.search(q, k=k, n_probe=n_probe)
                flat_results = flat.search(q, k=k)
                
                ivf_ids = {r.id for r in ivf_results}
                flat_ids = {r.id for r in flat_results}
                
                recall = len(ivf_ids & flat_ids) / k
                query_recalls.append(recall)
            
            recalls[n_probe] = np.mean(query_recalls)
        
        # Recall should generally increase with n_probe
        recall_values = list(recalls.values())
        assert recall_values[-1] >= recall_values[0]
        
        print("\n\nn_probe vs Recall:")
        for n_probe, recall in recalls.items():
            print(f"  n_probe={n_probe:2d}: recall={recall:.3f}")


class TestIVFIndexRemove:
    """IVF remove tests."""
    
    @pytest.fixture
    def populated_index(self):
        np.random.seed(42)
        dimension = 16
        
        index = IVFIndex(dimension=dimension, n_clusters=5, seed=42)
        training = np.random.randn(100, dimension).astype(np.float32)
        index.train(training)
        
        for i in range(100):
            index.add(f"vec{i}", np.random.randn(dimension).astype(np.float32))
        
        return index
    
    def test_remove(self, populated_index):
        """Test removing a vector."""
        assert "vec50" in populated_index
        
        result = populated_index.remove("vec50")
        
        assert result is True
        assert "vec50" not in populated_index
        assert populated_index.size == 99
    
    def test_remove_not_found(self, populated_index):
        """Test removing non-existent vector."""
        result = populated_index.remove("nonexistent")
        
        assert result is False
    
    def test_search_after_remove(self, populated_index):
        """Test that search works after removal."""
        # Remove some vectors
        for i in range(0, 50, 2):
            populated_index.remove(f"vec{i}")
        
        query = np.random.randn(16).astype(np.float32)
        results = populated_index.search(query, k=5)
        
        assert len(results) == 5
        # Removed vectors should not be in results
        for r in results:
            idx = int(r.id.replace("vec", ""))
            assert idx >= 50 or idx % 2 == 1


class TestIVFIndexGetContains:
    """Get and contains tests."""
    
    @pytest.fixture
    def index(self):
        np.random.seed(42)
        dimension = 10
        
        index = IVFIndex(dimension=dimension, n_clusters=3, seed=42)
        training = np.random.randn(50, dimension).astype(np.float32)
        index.train(training)
        
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


class TestIVFIndexSerialization:
    """Serialization tests."""
    
    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        np.random.seed(42)
        dimension = 16
        
        index = IVFIndex(dimension=dimension, n_clusters=5, n_probe=2, seed=42)
        training = np.random.randn(100, dimension).astype(np.float32)
        index.train(training)
        
        for i in range(50):
            index.add(f"vec{i}", np.random.randn(dimension).astype(np.float32), {"i": i})
        
        # Serialize
        data = index.to_dict()
        
        # Deserialize
        restored = IVFIndex.from_dict(data)
        
        assert restored.dimension == index.dimension
        assert restored.metric == index.metric
        assert restored.size == index.size
        assert restored.n_clusters == index.n_clusters
        assert restored.is_trained()
        
        # Check vectors match
        for id in ["vec0", "vec25", "vec49"]:
            orig_vec, orig_meta = index.get(id)
            rest_vec, rest_meta = restored.get(id)
            
            assert_array_almost_equal(orig_vec, rest_vec)
            assert orig_meta == rest_meta
        
        # Check search works
        query = np.random.randn(dimension).astype(np.float32)
        orig_results = index.search(query, k=5)
        rest_results = restored.search(query, k=5)
        
        # Results should match
        orig_ids = [r.id for r in orig_results]
        rest_ids = [r.id for r in rest_results]
        
        assert orig_ids == rest_ids


class TestIVFIndexStats:
    """Statistics tests."""
    
    def test_stats(self):
        """Test index statistics."""
        np.random.seed(42)
        dimension = 32
        
        index = IVFIndex(dimension=dimension, n_clusters=20, seed=42)
        training = np.random.randn(500, dimension).astype(np.float32)
        index.train(training)
        
        for i in range(1000):
            index.add(f"vec{i}", np.random.randn(dimension).astype(np.float32))
        
        stats = index.stats()
        
        assert stats.index_type == "ivf"
        assert stats.dimension == dimension
        assert stats.vector_count == 1000
        assert stats.is_trained is True
        assert stats.extra["n_clusters"] == 20
    
    def test_cluster_info(self):
        """Test cluster info."""
        np.random.seed(42)
        dimension = 16
        
        index = IVFIndex(dimension=dimension, n_clusters=10, seed=42)
        training = np.random.randn(200, dimension).astype(np.float32)
        index.train(training)
        
        for i in range(500):
            index.add(f"vec{i}", np.random.randn(dimension).astype(np.float32))
        
        info = index.get_cluster_info()
        
        assert info["n_clusters"] == 10
        assert info["total_vectors"] == 500
        assert "size_distribution" in info
        assert info["size_distribution"]["mean"] == pytest.approx(50, abs=20)


class TestIVFIndexClusterOperations:
    """Cluster operation tests."""
    
    @pytest.fixture
    def index(self):
        np.random.seed(42)
        dimension = 16
        
        index = IVFIndex(dimension=dimension, n_clusters=5, seed=42)
        training = np.random.randn(100, dimension).astype(np.float32)
        index.train(training)
        
        for i in range(50):
            index.add(f"vec{i}", np.random.randn(dimension).astype(np.float32))
        
        return index
    
    def test_assign_cluster(self, index):
        """Test cluster assignment."""
        vector = np.random.randn(16).astype(np.float32)
        
        cluster_id = index.assign_cluster(vector)
        
        assert 0 <= cluster_id < 5
    
    def test_get_centroid(self, index):
        """Test getting centroid."""
        centroid = index.get_centroid(0)
        
        assert len(centroid) == 16
    
    def test_get_vectors_in_cluster(self, index):
        """Test getting vectors in a cluster."""
        # At least one cluster should have vectors
        has_vectors = False
        
        for cluster_id in range(5):
            ids = index.get_vectors_in_cluster(cluster_id)
            if len(ids) > 0:
                has_vectors = True
                break
        
        assert has_vectors


class TestIVFIndexRebuild:
    """Rebuild tests."""
    
    def test_rebuild(self):
        """Test rebuilding index."""
        np.random.seed(42)
        dimension = 16
        
        index = IVFIndex(dimension=dimension, n_clusters=5, seed=42)
        training = np.random.randn(100, dimension).astype(np.float32)
        index.train(training)
        
        for i in range(100):
            index.add(f"vec{i}", np.random.randn(dimension).astype(np.float32))
        
        # Remove some
        for i in range(0, 50, 2):
            index.remove(f"vec{i}")
        
        original_size = index.size
        
        # Rebuild
        index.rebuild()
        
        assert index.size == original_size
        assert index.is_trained()
        
        # Search should work
        query = np.random.randn(dimension).astype(np.float32)
        results = index.search(query, k=5)
        assert len(results) == 5


class TestIVFIndexIteration:
    """Iteration tests."""
    
    @pytest.fixture
    def index(self):
        np.random.seed(42)
        dimension = 10
        
        index = IVFIndex(dimension=dimension, n_clusters=3, seed=42)
        training = np.random.randn(50, dimension).astype(np.float32)
        index.train(training)
        
        for i in range(20):
            index.add(f"vec{i}", np.random.randn(dimension).astype(np.float32))
        
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