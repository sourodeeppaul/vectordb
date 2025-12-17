"""
Integration tests for VectorDB query functionality.

Tests complex query scenarios including multi-filter queries,
aggregations, and query optimization.
"""

import pytest
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Any
from datetime import datetime, timedelta
import time

from vectordb.core.database import VectorDatabase
from vectordb.core.exceptions import (
    QueryError,
    InvalidFilterError,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for database storage."""
    path = tempfile.mkdtemp(prefix="vectordb_query_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def rich_dataset():
    """Generate a rich dataset for complex query testing."""
    np.random.seed(42)
    num_vectors = 1000
    dimension = 128
    
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    
    categories = ["electronics", "clothing", "books", "food", "toys"]
    brands = ["brand_a", "brand_b", "brand_c", "brand_d", None]
    
    base_date = datetime(2024, 1, 1)
    
    metadata = []
    for i in range(num_vectors):
        meta = {
            "product_id": f"prod_{i:04d}",
            "category": categories[i % len(categories)],
            "brand": brands[i % len(brands)],
            "price": round(10.0 + (i % 100) * 5.0 + np.random.rand() * 10, 2),
            "rating": round(1.0 + np.random.rand() * 4, 1),
            "reviews_count": int(np.random.exponential(50)),
            "in_stock": i % 3 != 0,
            "created_at": (base_date + timedelta(days=i % 365)).isoformat(),
            "tags": [f"tag_{j}" for j in range(i % 5)],
            "nested": {
                "level1": {
                    "value": i % 10,
                    "name": f"nested_{i}"
                }
            }
        }
        metadata.append(meta)
    
    ids = [f"vec_{i:04d}" for i in range(num_vectors)]
    
    return vectors, metadata, ids


@pytest.fixture
def populated_database(temp_db_path, rich_dataset):
    """Create a database populated with rich test data."""
    vectors, metadata, ids = rich_dataset
    
    db = VectorDatabase(storage_path=temp_db_path)
    collection = db.create_collection(
        "products",
        dimension=128,
        metric="cosine"
    )
    collection.add(vectors, ids=ids, metadata=metadata)
    
    yield db, collection
    
    db.close()


class TestBasicFilters:
    """Test basic filter operations."""

    def test_equality_filter(self, populated_database):
        """Test exact equality filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"category": "electronics"}
        )
        
        assert len(results) > 0
        assert all(r["metadata"]["category"] == "electronics" for r in results)

    def test_boolean_filter(self, populated_database):
        """Test boolean field filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"in_stock": True}
        )
        
        assert all(r["metadata"]["in_stock"] is True for r in results)

    def test_null_filter(self, populated_database):
        """Test filtering for null/None values."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        # Find items without brand
        results = collection.search(
            query,
            k=50,
            filter={"brand": None}
        )
        
        assert all(r["metadata"]["brand"] is None for r in results)

    def test_not_null_filter(self, populated_database):
        """Test filtering for non-null values."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"brand": {"$ne": None}}
        )
        
        assert all(r["metadata"]["brand"] is not None for r in results)


class TestComparisonFilters:
    """Test comparison operators in filters."""

    def test_greater_than_filter(self, populated_database):
        """Test $gt filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"price": {"$gt": 200}}
        )
        
        assert all(r["metadata"]["price"] > 200 for r in results)

    def test_greater_than_or_equal_filter(self, populated_database):
        """Test $gte filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"rating": {"$gte": 4.0}}
        )
        
        assert all(r["metadata"]["rating"] >= 4.0 for r in results)

    def test_less_than_filter(self, populated_database):
        """Test $lt filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"price": {"$lt": 50}}
        )
        
        assert all(r["metadata"]["price"] < 50 for r in results)

    def test_less_than_or_equal_filter(self, populated_database):
        """Test $lte filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"reviews_count": {"$lte": 10}}
        )
        
        assert all(r["metadata"]["reviews_count"] <= 10 for r in results)

    def test_not_equal_filter(self, populated_database):
        """Test $ne filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"category": {"$ne": "electronics"}}
        )
        
        assert all(r["metadata"]["category"] != "electronics" for r in results)

    def test_range_filter(self, populated_database):
        """Test combined range filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={
                "price": {"$gte": 50, "$lte": 150}
            }
        )
        
        for r in results:
            assert 50 <= r["metadata"]["price"] <= 150


class TestCollectionFilters:
    """Test collection-based filter operators."""

    def test_in_filter(self, populated_database):
        """Test $in filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"category": {"$in": ["electronics", "books"]}}
        )
        
        for r in results:
            assert r["metadata"]["category"] in ["electronics", "books"]

    def test_not_in_filter(self, populated_database):
        """Test $nin filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"category": {"$nin": ["electronics", "toys"]}}
        )
        
        for r in results:
            assert r["metadata"]["category"] not in ["electronics", "toys"]

    def test_contains_filter(self, populated_database):
        """Test $contains filter for array fields."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"tags": {"$contains": "tag_1"}}
        )
        
        for r in results:
            assert "tag_1" in r["metadata"]["tags"]


class TestLogicalFilters:
    """Test logical operators in filters."""

    def test_and_filter(self, populated_database):
        """Test $and filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={
                "$and": [
                    {"category": "electronics"},
                    {"in_stock": True},
                    {"rating": {"$gte": 3.0}}
                ]
            }
        )
        
        for r in results:
            assert r["metadata"]["category"] == "electronics"
            assert r["metadata"]["in_stock"] is True
            assert r["metadata"]["rating"] >= 3.0

    def test_or_filter(self, populated_database):
        """Test $or filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={
                "$or": [
                    {"category": "electronics"},
                    {"category": "books"}
                ]
            }
        )
        
        for r in results:
            assert r["metadata"]["category"] in ["electronics", "books"]

    def test_not_filter(self, populated_database):
        """Test $not filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={
                "$not": {"category": "electronics"}
            }
        )
        
        assert all(r["metadata"]["category"] != "electronics" for r in results)

    def test_complex_nested_filter(self, populated_database):
        """Test complex nested logical filters."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={
                "$and": [
                    {
                        "$or": [
                            {"category": "electronics"},
                            {"category": "toys"}
                        ]
                    },
                    {"price": {"$lt": 100}},
                    {
                        "$or": [
                            {"rating": {"$gte": 4.0}},
                            {"reviews_count": {"$gte": 100}}
                        ]
                    }
                ]
            }
        )
        
        for r in results:
            meta = r["metadata"]
            assert meta["category"] in ["electronics", "toys"]
            assert meta["price"] < 100
            assert meta["rating"] >= 4.0 or meta["reviews_count"] >= 100


class TestNestedFieldFilters:
    """Test filtering on nested metadata fields."""

    def test_nested_field_equality(self, populated_database):
        """Test equality filter on nested field."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"nested.level1.value": 5}
        )
        
        for r in results:
            assert r["metadata"]["nested"]["level1"]["value"] == 5

    def test_nested_field_comparison(self, populated_database):
        """Test comparison filter on nested field."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"nested.level1.value": {"$gte": 7}}
        )
        
        for r in results:
            assert r["metadata"]["nested"]["level1"]["value"] >= 7


class TestTextFilters:
    """Test text-based filter operations."""

    def test_string_prefix_filter(self, populated_database):
        """Test string prefix filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"product_id": {"$startswith": "prod_00"}}
        )
        
        for r in results:
            assert r["metadata"]["product_id"].startswith("prod_00")

    def test_string_contains_filter(self, populated_database):
        """Test string contains filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"brand": {"$contains": "brand_"}}
        )
        
        for r in results:
            assert r["metadata"]["brand"] is not None
            assert "brand_" in r["metadata"]["brand"]

    def test_regex_filter(self, populated_database):
        """Test regex-based filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=50,
            filter={"product_id": {"$regex": r"prod_0[0-1]\d\d"}}
        )
        
        for r in results:
            import re
            assert re.match(r"prod_0[0-1]\d\d", r["metadata"]["product_id"])


class TestQueryPerformance:
    """Test query performance characteristics."""

    def test_filter_reduces_search_space(self, populated_database):
        """Test that filters properly reduce search candidates."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        # Measure unfiltered search
        start = time.time()
        unfiltered_results = collection.search(query, k=10)
        unfiltered_time = time.time() - start
        
        # Measure filtered search (should check fewer vectors)
        start = time.time()
        filtered_results = collection.search(
            query,
            k=10,
            filter={"category": "electronics"}  # ~20% of data
        )
        filtered_time = time.time() - start
        
        # Verify filter was applied
        assert all(r["metadata"]["category"] == "electronics" for r in filtered_results)

    def test_highly_selective_filter(self, populated_database):
        """Test performance with highly selective filter."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        # Very selective filter - only matches few items
        results = collection.search(
            query,
            k=100,
            filter={
                "$and": [
                    {"category": "electronics"},
                    {"rating": {"$gte": 4.5}},
                    {"price": {"$lt": 30}},
                    {"in_stock": True}
                ]
            }
        )
        
        # Should return fewer than k if not enough matches
        assert len(results) <= 100

    def test_batch_queries_efficiency(self, populated_database):
        """Test batch query execution."""
        db, collection = populated_database
        
        queries = np.random.randn(10, 128).astype(np.float32)
        
        # Batch query
        start = time.time()
        batch_results = collection.search_batch(queries, k=10)
        batch_time = time.time() - start
        
        # Individual queries
        start = time.time()
        individual_results = [collection.search(q, k=10) for q in queries]
        individual_time = time.time() - start
        
        # Batch should not be significantly slower
        assert batch_time <= individual_time * 1.5
        
        # Results should be equivalent
        assert len(batch_results) == len(individual_results)


class TestQueryEdgeCases:
    """Test edge cases and error handling in queries."""

    def test_empty_result_set(self, populated_database):
        """Test query that matches nothing."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=10,
            filter={"category": "nonexistent_category"}
        )
        
        assert len(results) == 0

    def test_invalid_filter_field(self, populated_database):
        """Test filter on non-existent field."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        # Should handle gracefully - no matches
        results = collection.search(
            query,
            k=10,
            filter={"nonexistent_field": "value"}
        )
        
        # Either returns empty or raises error
        assert isinstance(results, list)

    def test_invalid_filter_syntax(self, populated_database):
        """Test invalid filter syntax raises error."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        with pytest.raises((InvalidFilterError, QueryError, ValueError)):
            collection.search(
                query,
                k=10,
                filter={"price": {"$invalid_operator": 100}}
            )

    def test_type_mismatch_in_filter(self, populated_database):
        """Test filter with mismatched types."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        # String compared to number field
        results = collection.search(
            query,
            k=10,
            filter={"price": "not_a_number"}
        )
        
        # Should return no matches (type mismatch)
        assert len(results) == 0

    def test_very_deep_nesting(self, populated_database):
        """Test filter on very deeply nested field."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        # Deeper than actual data
        results = collection.search(
            query,
            k=10,
            filter={"nested.level1.level2.level3.value": 5}
        )
        
        # Should return no matches
        assert len(results) == 0


class TestQueryWithDifferentIndexTypes:
    """Test queries work correctly with different index types."""

    @pytest.fixture
    def multi_index_db(self, temp_db_path, rich_dataset):
        """Create database with multiple index types."""
        vectors, metadata, ids = rich_dataset
        
        db = VectorDatabase(storage_path=temp_db_path)
        
        # Create collections with different index types
        for index_type in ["flat", "ivf", "hnsw"]:
            collection = db.create_collection(
                f"collection_{index_type}",
                dimension=128,
                index_type=index_type
            )
            collection.add(vectors[:200], ids=ids[:200], metadata=metadata[:200])
        
        yield db
        
        db.close()

    def test_filter_works_with_all_indexes(self, multi_index_db):
        """Test filters work with all index types."""
        query = np.random.randn(128).astype(np.float32)
        
        for index_type in ["flat", "ivf", "hnsw"]:
            collection = multi_index_db.get_collection(f"collection_{index_type}")
            
            results = collection.search(
                query,
                k=20,
                filter={"category": "electronics"}
            )
            
            assert all(r["metadata"]["category"] == "electronics" for r in results)

    def test_complex_filter_with_all_indexes(self, multi_index_db):
        """Test complex filters with all index types."""
        query = np.random.randn(128).astype(np.float32)
        
        filter_config = {
            "$and": [
                {"category": {"$in": ["electronics", "books"]}},
                {"price": {"$gte": 50}},
                {"in_stock": True}
            ]
        }
        
        for index_type in ["flat", "ivf", "hnsw"]:
            collection = multi_index_db.get_collection(f"collection_{index_type}")
            
            results = collection.search(query, k=20, filter=filter_config)
            
            for r in results:
                meta = r["metadata"]
                assert meta["category"] in ["electronics", "books"]
                assert meta["price"] >= 50
                assert meta["in_stock"] is True


class TestQueryProjection:
    """Test query result projection (selecting specific fields)."""

    def test_include_specific_fields(self, populated_database):
        """Test including only specific metadata fields."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=10,
            include_metadata=["category", "price"]
        )
        
        for r in results:
            # Should only have specified fields
            assert "category" in r["metadata"]
            assert "price" in r["metadata"]
            # Other fields might be excluded depending on implementation

    def test_exclude_vectors(self, populated_database):
        """Test excluding vectors from results."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results = collection.search(
            query,
            k=10,
            include_vectors=False
        )
        
        for r in results:
            assert "vector" not in r or r.get("vector") is None


class TestQueryStats:
    """Test query statistics and explain functionality."""

    def test_query_returns_stats(self, populated_database):
        """Test that queries can return execution stats."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        results, stats = collection.search(
            query,
            k=10,
            return_stats=True
        )
        
        assert "execution_time_ms" in stats
        assert "vectors_scanned" in stats

    def test_explain_query(self, populated_database):
        """Test query explain functionality."""
        db, collection = populated_database
        query = np.random.randn(128).astype(np.float32)
        
        explanation = collection.explain_search(
            query,
            k=10,
            filter={"category": "electronics"}
        )
        
        assert "query_plan" in explanation or "plan" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])