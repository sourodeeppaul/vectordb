"""
VectorDB - A vector database for similarity search.

Example:
    >>> from vectordb import VectorDB
    >>> import numpy as np
    >>> 
    >>> # Create database and collection
    >>> db = VectorDB()
    >>> collection = db.create_collection("docs", dimension=128)
    >>> 
    >>> # Add vectors
    >>> collection.add("doc1", np.random.randn(128), {"type": "article"})
    >>> 
    >>> # Search
    >>> results = collection.search(np.random.randn(128), k=5)
"""

from .core import (
    # Main classes
    VectorDB,
    Collection,
    VectorRecord,
    VectorBatch,
    SearchResult,
    # Config
    CollectionConfig,
    CollectionStats,
    IndexType,
    # Exceptions
    VectorDBError,
    CollectionNotFoundError,
    CollectionExistsError,
    VectorNotFoundError,
    VectorExistsError,
    DimensionMismatchError,
    ValidationError,
)

from .distance import (
    # Metrics
    euclidean,
    cosine_distance,
    cosine_similarity,
    dot_product,
    manhattan,
    # Registry
    get_metric,
    get_metric_fn,
    list_metrics,
    DistanceMetric,
)

__version__ = "0.1.0"
__author__ = "VectorDB Team"

__all__ = [
    # Main classes
    "VectorDB",
    "Collection",
    "VectorRecord",
    "VectorBatch",
    "SearchResult",
    # Config
    "CollectionConfig",
    "CollectionStats",
    "IndexType",
    # Exceptions
    "VectorDBError",
    "CollectionNotFoundError",
    "CollectionExistsError",
    "VectorNotFoundError",
    "VectorExistsError",
    "DimensionMismatchError",
    "ValidationError",
    # Distance functions
    "euclidean",
    "cosine_distance",
    "cosine_similarity",
    "dot_product",
    "manhattan",
    "get_metric",
    "get_metric_fn",
    "list_metrics",
    "DistanceMetric",
]