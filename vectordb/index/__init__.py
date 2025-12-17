"""
Index implementations for VectorDB.

Available Indices:
    - FlatIndex: Brute-force exact search (100% recall)
    - HNSWIndex: HNSW graph index (fast approximate search)
    - IVFIndex: Inverted file index (clustering-based search)

Example:
    >>> from vectordb.index import FlatIndex, HNSWIndex, IVFIndex, create_index
    >>> 
    >>> # Flat for small datasets
    >>> flat = FlatIndex(dimension=128)
    >>> 
    >>> # HNSW for large datasets
    >>> hnsw = HNSWIndex(dimension=128, M=16, ef_search=50)
    >>> 
    >>> # IVF for very large datasets
    >>> ivf = IVFIndex(dimension=128, n_clusters=100, n_probe=10)
    >>> ivf.train(training_vectors)
"""

from .base import (
    BaseIndex,
    IndexConfig,
    IndexStats,
    SearchResult,
    IndexType,
)

from .flat import FlatIndex, FlatIndexConfig
from .hnsw import HNSWIndex, HNSWConfig
from .ivf import IVFIndex, IVFConfig

__all__ = [
    # Base
    "BaseIndex",
    "IndexConfig",
    "IndexStats",
    "SearchResult",
    "IndexType",
    # Flat
    "FlatIndex",
    "FlatIndexConfig",
    # HNSW
    "HNSWIndex",
    "HNSWConfig",
    # IVF
    "IVFIndex",
    "IVFConfig",
    # Factory
    "create_index",
]


def create_index(
    index_type: str,
    dimension: int,
    metric: str = "euclidean",
    **kwargs
) -> BaseIndex:
    """
    Factory function to create an index.
    
    Args:
        index_type: Type of index ("flat", "hnsw", "ivf")
        dimension: Vector dimension
        metric: Distance metric
        **kwargs: Index-specific parameters
        
    Returns:
        Index instance
        
    Example:
        >>> # Flat index for small datasets
        >>> index = create_index("flat", dimension=128)
        >>> 
        >>> # HNSW for large datasets
        >>> index = create_index("hnsw", dimension=128, M=32, ef_search=100)
        >>> 
        >>> # IVF for very large datasets (requires training)
        >>> index = create_index("ivf", dimension=128, n_clusters=1000, n_probe=50)
    """
    index_type = index_type.lower()
    
    if index_type == "flat":
        return FlatIndex(dimension=dimension, metric=metric, **kwargs)
    elif index_type == "hnsw":
        return HNSWIndex(dimension=dimension, metric=metric, **kwargs)
    elif index_type == "ivf":
        return IVFIndex(dimension=dimension, metric=metric, **kwargs)
    else:
        raise ValueError(
            f"Unknown index type: {index_type}. "
            f"Available: flat, hnsw, ivf"
        )


def recommend_index(
    n_vectors: int,
    dimension: int,
    priority: str = "balanced",
) -> Dict[str, Any]:
    """
    Recommend index type based on dataset characteristics.
    
    Args:
        n_vectors: Expected number of vectors
        dimension: Vector dimension
        priority: "speed", "recall", or "balanced"
        
    Returns:
        Dictionary with recommended index type and parameters
        
    Example:
        >>> rec = recommend_index(1000000, 128, priority="speed")
        >>> print(rec)
        {'index_type': 'ivf', 'params': {'n_clusters': 1000, 'n_probe': 10}}
    """
    if n_vectors < 10000:
        # Small dataset: use Flat
        return {
            "index_type": "flat",
            "params": {},
            "reason": "Dataset small enough for exact search",
        }
    
    elif n_vectors < 100000:
        # Medium dataset: use HNSW
        if priority == "speed":
            return {
                "index_type": "hnsw",
                "params": {"M": 12, "ef_construction": 100, "ef_search": 32},
                "reason": "HNSW optimized for speed",
            }
        elif priority == "recall":
            return {
                "index_type": "hnsw",
                "params": {"M": 32, "ef_construction": 400, "ef_search": 200},
                "reason": "HNSW optimized for recall",
            }
        else:
            return {
                "index_type": "hnsw",
                "params": {"M": 16, "ef_construction": 200, "ef_search": 64},
                "reason": "HNSW with balanced parameters",
            }
    
    else:
        # Large dataset: use IVF or HNSW
        n_clusters = int(4 * np.sqrt(n_vectors))
        
        if priority == "speed":
            return {
                "index_type": "ivf",
                "params": {"n_clusters": n_clusters, "n_probe": max(1, n_clusters // 100)},
                "reason": "IVF optimized for speed on large datasets",
            }
        elif priority == "recall":
            return {
                "index_type": "hnsw",
                "params": {"M": 32, "ef_construction": 400, "ef_search": 200},
                "reason": "HNSW provides better recall for large datasets",
            }
        else:
            return {
                "index_type": "ivf",
                "params": {"n_clusters": n_clusters, "n_probe": max(1, n_clusters // 20)},
                "reason": "IVF with balanced parameters for large datasets",
            }


# Import numpy for recommend_index
import numpy as np