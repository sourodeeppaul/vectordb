"""
Unit tests for distance metrics.
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from vectordb.distance import (
    # Single vector functions
    euclidean,
    euclidean_squared,
    cosine_distance,
    cosine_similarity,
    dot_product,
    negative_dot_product,
    manhattan,
    chebyshev,
    hamming,
    # Batch functions
    pairwise_euclidean,
    pairwise_cosine,
    pairwise_dot,
    pairwise_manhattan,
    query_distances,
    # Registry
    get_metric,
    get_metric_fn,
    list_metrics,
    register_metric,
    DistanceMetric,
    # Batch calculator
    BatchDistanceCalculator,
    compute_top_k,
)


class TestEuclideanDistance:
    """Tests for Euclidean distance."""
    
    def test_zero_distance(self):
        """Same vectors should have zero distance."""
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean(a, a) == 0.0
    
    def test_known_distance(self):
        """Test with known distance (3-4-5 triangle)."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert_almost_equal(euclidean(a, b), 5.0)
    
    def test_symmetry(self):
        """Distance should be symmetric."""
        a = np.random.randn(10).astype(np.float32)
        b = np.random.randn(10).astype(np.float32)
        assert_almost_equal(euclidean(a, b), euclidean(b, a))
    
    def test_non_negative(self):
        """Distance should be non-negative."""
        a = np.random.randn(10).astype(np.float32)
        b = np.random.randn(10).astype(np.float32)
        assert euclidean(a, b) >= 0
    
    def test_squared_vs_regular(self):
        """Squared should be square of regular."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert_almost_equal(euclidean_squared(a, b), euclidean(a, b) ** 2)


class TestCosineDistance:
    """Tests for Cosine distance/similarity."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1, distance 0."""
        a = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(cosine_similarity(a, a), 1.0)
        assert_almost_equal(cosine_distance(a, a), 0.0)
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1, distance 2."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert_almost_equal(cosine_similarity(a, b), -1.0)
        assert_almost_equal(cosine_distance(a, b), 2.0)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0, distance 1."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert_almost_equal(cosine_similarity(a, b), 0.0)
        assert_almost_equal(cosine_distance(a, b), 1.0)
    
    def test_scale_invariance(self):
        """Cosine should be scale-invariant."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # Same direction, different magnitude
        assert_almost_equal(cosine_similarity(a, b), 1.0)
    
    def test_range(self):
        """Similarity should be in [-1, 1], distance in [0, 2]."""
        for _ in range(100):
            a = np.random.randn(10).astype(np.float32)
            b = np.random.randn(10).astype(np.float32)
            
            sim = cosine_similarity(a, b)
            dist = cosine_distance(a, b)
            
            assert -1.0 <= sim <= 1.0
            assert 0.0 <= dist <= 2.0


class TestDotProduct:
    """Tests for dot product."""
    
    def test_basic(self):
        """Test basic dot product."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert_almost_equal(dot_product(a, b), 32.0)  # 1*4 + 2*5 + 3*6
    
    def test_orthogonal(self):
        """Orthogonal vectors have zero dot product."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert_almost_equal(dot_product(a, b), 0.0)
    
    def test_negative_dot_product(self):
        """Negative dot product for distance."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert_almost_equal(negative_dot_product(a, b), -11.0)
    
    def test_normalized_equals_cosine(self):
        """For normalized vectors, dot product equals cosine similarity."""
        a = np.random.randn(10).astype(np.float32)
        b = np.random.randn(10).astype(np.float32)
        
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        
        assert_almost_equal(dot_product(a_norm, b_norm), cosine_similarity(a, b), decimal=5)


class TestManhattanDistance:
    """Tests for Manhattan distance."""
    
    def test_zero_distance(self):
        """Same vectors should have zero distance."""
        a = np.array([1.0, 2.0, 3.0])
        assert manhattan(a, a) == 0.0
    
    def test_known_distance(self):
        """Test with known distance."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert_almost_equal(manhattan(a, b), 7.0)  # 3 + 4
    
    def test_symmetry(self):
        """Distance should be symmetric."""
        a = np.random.randn(10).astype(np.float32)
        b = np.random.randn(10).astype(np.float32)
        assert_almost_equal(manhattan(a, b), manhattan(b, a))


class TestChebyshevDistance:
    """Tests for Chebyshev distance."""
    
    def test_zero_distance(self):
        """Same vectors should have zero distance."""
        a = np.array([1.0, 2.0, 3.0])
        assert chebyshev(a, a) == 0.0
    
    def test_known_distance(self):
        """Test with known distance."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert_almost_equal(chebyshev(a, b), 4.0)  # max(3, 4)


class TestHammingDistance:
    """Tests for Hamming distance."""
    
    def test_identical(self):
        """Identical vectors have zero Hamming distance."""
        a = np.array([1, 0, 1, 1, 0])
        assert hamming(a, a) == 0.0
    
    def test_known_distance(self):
        """Test with known distance."""
        a = np.array([1, 0, 1, 1, 0])
        b = np.array([1, 1, 1, 0, 0])
        assert hamming(a, b) == 2.0  # Positions 1 and 3 differ


class TestPairwiseDistances:
    """Tests for pairwise distance functions."""
    
    def test_pairwise_euclidean_self(self):
        """Diagonal should be zero for self-distance."""
        X = np.random.randn(10, 5).astype(np.float32)
        D = pairwise_euclidean(X)
        
        assert D.shape == (10, 10)
        assert_array_almost_equal(np.diag(D), np.zeros(10))
    
    def test_pairwise_euclidean_symmetry(self):
        """Distance matrix should be symmetric."""
        X = np.random.randn(10, 5).astype(np.float32)
        D = pairwise_euclidean(X)
        
        assert_array_almost_equal(D, D.T)
    
    def test_pairwise_euclidean_xy(self):
        """Test X vs Y pairwise distances."""
        X = np.random.randn(10, 5).astype(np.float32)
        Y = np.random.randn(20, 5).astype(np.float32)
        D = pairwise_euclidean(X, Y)
        
        assert D.shape == (10, 20)
        
        # Verify a few entries
        for i in range(min(3, len(X))):
            for j in range(min(3, len(Y))):
                expected = euclidean(X[i], Y[j])
                assert_almost_equal(D[i, j], expected, decimal=5)
    
    def test_pairwise_cosine(self):
        """Test pairwise cosine distances."""
        X = np.random.randn(10, 5).astype(np.float32)
        D = pairwise_cosine(X)
        
        assert D.shape == (10, 10)
        assert_array_almost_equal(np.diag(D), np.zeros(10))  # Self-distance = 0
        assert np.all(D >= -1e-6)  # All distances >= 0
        assert np.all(D <= 2.0 + 1e-6)  # All distances <= 2
    
    def test_pairwise_dot(self):
        """Test pairwise dot product distances."""
        X = np.random.randn(10, 5).astype(np.float32)
        D = pairwise_dot(X)
        
        assert D.shape == (10, 10)
        
        # Verify: negative dot products
        for i in range(3):
            for j in range(3):
                expected = -np.dot(X[i], X[j])
                assert_almost_equal(D[i, j], expected, decimal=5)


class TestQueryDistances:
    """Tests for query-to-collection distances."""
    
    def test_euclidean(self):
        """Test Euclidean query distances."""
        query = np.array([0.0, 0.0, 0.0])
        collection = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        
        distances = query_distances(query, collection, "euclidean")
        
        assert_array_almost_equal(distances, [1.0, 2.0, 3.0])
    
    def test_cosine(self):
        """Test cosine query distances."""
        query = np.array([1.0, 0.0])
        collection = np.array([
            [1.0, 0.0],   # Same direction
            [0.0, 1.0],   # Orthogonal
            [-1.0, 0.0],  # Opposite
        ])
        
        distances = query_distances(query, collection, "cosine")
        
        assert_array_almost_equal(distances, [0.0, 1.0, 2.0])
    
    def test_output_shape(self):
        """Test output shape matches collection."""
        query = np.random.randn(10).astype(np.float32)
        collection = np.random.randn(100, 10).astype(np.float32)
        
        for metric in ["euclidean", "cosine", "dot", "manhattan"]:
            distances = query_distances(query, collection, metric)
            assert distances.shape == (100,)


class TestMetricRegistry:
    """Tests for metric registry."""
    
    def test_list_metrics(self):
        """Test listing available metrics."""
        metrics = list_metrics()
        
        assert "euclidean" in metrics
        assert "cosine" in metrics
        assert "manhattan" in metrics
    
    def test_get_metric(self):
        """Test getting metric info."""
        info = get_metric("euclidean")
        
        assert info.name == "euclidean"
        assert info.is_similarity is False
        assert callable(info.function)
    
    def test_get_metric_fn(self):
        """Test getting metric function."""
        fn = get_metric_fn("euclidean")
        
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        
        assert_almost_equal(fn(a, b), 5.0)
    
    def test_alias(self):
        """Test metric aliases."""
        # l2 is alias for euclidean
        fn_l2 = get_metric_fn("l2")
        fn_euclidean = get_metric_fn("euclidean")
        
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        
        assert fn_l2(a, b) == fn_euclidean(a, b)
    
    def test_unknown_metric(self):
        """Test error for unknown metric."""
        with pytest.raises(KeyError):
            get_metric("unknown_metric")
    
    def test_register_custom(self):
        """Test registering custom metric."""
        def custom_dist(a, b):
            return np.sum(a * b)
        
        register_metric(
            "test_custom",
            custom_dist,
            description="Test custom metric"
        )
        
        fn = get_metric_fn("test_custom")
        assert fn is custom_dist


class TestBatchDistanceCalculator:
    """Tests for BatchDistanceCalculator."""
    
    def test_compute_all(self):
        """Test computing all distances."""
        query = np.random.randn(10).astype(np.float32)
        collection = np.random.randn(100, 10).astype(np.float32)
        
        calc = BatchDistanceCalculator(metric="euclidean")
        distances = calc.compute(query, collection)
        
        assert distances.shape == (100,)
    
    def test_top_k(self):
        """Test finding top-k neighbors."""
        query = np.zeros(10, dtype=np.float32)
        
        # Create collection where first 5 vectors are closest
        collection = np.random.randn(100, 10).astype(np.float32)
        for i in range(5):
            collection[i] = np.random.randn(10).astype(np.float32) * 0.1
        
        calc = BatchDistanceCalculator(metric="euclidean")
        indices, distances = calc.top_k(query, collection, k=5)
        
        assert len(indices) == 5
        assert len(distances) == 5
        
        # Distances should be sorted
        assert np.all(distances[:-1] <= distances[1:])
    
    def test_top_k_larger_than_collection(self):
        """Test top-k when k > collection size."""
        query = np.random.randn(10).astype(np.float32)
        collection = np.random.randn(5, 10).astype(np.float32)
        
        calc = BatchDistanceCalculator(metric="euclidean")
        indices, distances = calc.top_k(query, collection, k=10)
        
        assert len(indices) == 5  # Capped at collection size
    
    def test_chunked_processing(self):
        """Test chunked processing gives same results."""
        query = np.random.randn(10).astype(np.float32)
        collection = np.random.randn(1000, 10).astype(np.float32)
        
        # Small chunks
        calc_small = BatchDistanceCalculator(metric="euclidean", chunk_size=100)
        dist_small = calc_small.compute(query, collection)
        
        # Large chunks (no chunking)
        calc_large = BatchDistanceCalculator(metric="euclidean", chunk_size=10000)
        dist_large = calc_large.compute(query, collection)
        
        assert_array_almost_equal(dist_small, dist_large)


class TestComputeTopK:
    """Tests for compute_top_k convenience function."""
    
    def test_basic(self):
        """Test basic top-k computation."""
        query = np.array([0.0, 0.0])
        collection = np.array([
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 4.0],
        ])
        
        indices, distances = compute_top_k(query, collection, k=2)
        
        assert len(indices) == 2
        assert 0 in indices  # Distance 1.0
        assert 1 in indices  # Distance 2.0
    
    def test_sorted_output(self):
        """Test that output is sorted by distance."""
        query = np.random.randn(10).astype(np.float32)
        collection = np.random.randn(100, 10).astype(np.float32)
        
        indices, distances = compute_top_k(query, collection, k=10)
        
        # Verify sorted
        assert np.all(distances[:-1] <= distances[1:])
        
        # Verify indices match distances
        for i, idx in enumerate(indices):
            expected = euclidean(query, collection[idx])
            assert_almost_equal(distances[i], expected, decimal=5)