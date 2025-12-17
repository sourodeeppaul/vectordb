"""
Distance metric registry and factory.

Provides a unified interface for accessing distance functions
by name and managing custom metrics.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .metrics import (
    euclidean,
    euclidean_squared,
    cosine_distance,
    dot_product,
    negative_dot_product,
    manhattan,
    chebyshev,
    hamming,
    minkowski,
    angular_distance,
    query_distances,
    pairwise_euclidean,
    pairwise_euclidean_squared,
    pairwise_cosine,
    pairwise_dot,
    pairwise_manhattan,
)


# Type aliases
Vector = NDArray[np.floating]
DistanceFunction = Callable[[Vector, Vector], float]
BatchDistanceFunction = Callable[[NDArray, Optional[NDArray]], NDArray]


class DistanceMetric(str, Enum):
    """Enumeration of built-in distance metrics."""
    
    EUCLIDEAN = "euclidean"
    EUCLIDEAN_SQUARED = "euclidean_squared"
    COSINE = "cosine"
    DOT = "dot"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    HAMMING = "hamming"
    ANGULAR = "angular"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class MetricInfo:
    """Information about a distance metric."""
    
    name: str
    function: DistanceFunction
    batch_function: Optional[BatchDistanceFunction]
    is_similarity: bool  # True if larger values = more similar
    min_value: float
    max_value: Optional[float]  # None if unbounded
    description: str
    requires_normalization: bool = False
    
    def __repr__(self) -> str:
        return f"MetricInfo(name='{self.name}', is_similarity={self.is_similarity})"


# =============================================================================
# METRIC REGISTRY
# =============================================================================

class MetricRegistry:
    """
    Registry for distance metrics.
    
    Allows looking up metrics by name and registering custom metrics.
    """
    
    def __init__(self):
        self._metrics: Dict[str, MetricInfo] = {}
        self._aliases: Dict[str, str] = {}
        self._register_builtins()
    
    def _register_builtins(self) -> None:
        """Register built-in distance metrics."""
        
        # Euclidean distance
        self.register(
            MetricInfo(
                name="euclidean",
                function=euclidean,
                batch_function=pairwise_euclidean,
                is_similarity=False,
                min_value=0.0,
                max_value=None,
                description="Euclidean (L2) distance"
            ),
            aliases=["l2", "euclidean_distance"]
        )
        
        # Squared Euclidean distance
        self.register(
            MetricInfo(
                name="euclidean_squared",
                function=euclidean_squared,
                batch_function=pairwise_euclidean_squared,
                is_similarity=False,
                min_value=0.0,
                max_value=None,
                description="Squared Euclidean distance (faster, same ordering)"
            ),
            aliases=["l2_squared", "sqeuclidean"]
        )
        
        # Cosine distance
        self.register(
            MetricInfo(
                name="cosine",
                function=cosine_distance,
                batch_function=pairwise_cosine,
                is_similarity=False,
                min_value=0.0,
                max_value=2.0,
                description="Cosine distance (1 - cosine similarity)"
            ),
            aliases=["cosine_distance"]
        )
        
        # Dot product (as distance: negative)
        self.register(
            MetricInfo(
                name="dot",
                function=negative_dot_product,
                batch_function=pairwise_dot,
                is_similarity=False,
                min_value=None,
                max_value=None,
                description="Negative dot product (for normalized vectors)",
                requires_normalization=True
            ),
            aliases=["inner_product", "ip"]
        )
        
        # Manhattan distance
        self.register(
            MetricInfo(
                name="manhattan",
                function=manhattan,
                batch_function=pairwise_manhattan,
                is_similarity=False,
                min_value=0.0,
                max_value=None,
                description="Manhattan (L1) distance"
            ),
            aliases=["l1", "cityblock", "taxicab"]
        )
        
        # Chebyshev distance
        self.register(
            MetricInfo(
                name="chebyshev",
                function=chebyshev,
                batch_function=None,  # No optimized batch version
                is_similarity=False,
                min_value=0.0,
                max_value=None,
                description="Chebyshev (L∞) distance"
            ),
            aliases=["linf", "chessboard"]
        )
        
        # Hamming distance
        self.register(
            MetricInfo(
                name="hamming",
                function=hamming,
                batch_function=None,
                is_similarity=False,
                min_value=0.0,
                max_value=None,
                description="Hamming distance (for binary vectors)"
            ),
            aliases=[]
        )
        
        # Angular distance
        self.register(
            MetricInfo(
                name="angular",
                function=angular_distance,
                batch_function=None,
                is_similarity=False,
                min_value=0.0,
                max_value=np.pi,
                description="Angular distance (angle in radians)"
            ),
            aliases=[]
        )
    
    def register(
        self, 
        info: MetricInfo,
        aliases: Optional[List[str]] = None
    ) -> None:
        """
        Register a distance metric.
        
        Args:
            info: MetricInfo object
            aliases: Optional list of alternative names
        """
        self._metrics[info.name] = info
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = info.name
    
    def get(self, name: str) -> MetricInfo:
        """
        Get metric info by name.
        
        Args:
            name: Metric name or alias
            
        Returns:
            MetricInfo object
            
        Raises:
            KeyError: If metric not found
        """
        # Check aliases first
        canonical = self._aliases.get(name, name)
        
        if canonical not in self._metrics:
            available = list(self._metrics.keys())
            raise KeyError(
                f"Unknown metric: '{name}'. Available: {available}"
            )
        
        return self._metrics[canonical]
    
    def get_function(self, name: str) -> DistanceFunction:
        """Get the distance function for a metric."""
        return self.get(name).function
    
    def get_batch_function(self, name: str) -> Optional[BatchDistanceFunction]:
        """Get the batch distance function for a metric."""
        return self.get(name).batch_function
    
    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        return list(self._metrics.keys())
    
    def list_all(self) -> Dict[str, MetricInfo]:
        """Get all registered metrics with their info."""
        return self._metrics.copy()
    
    def is_similarity(self, name: str) -> bool:
        """Check if a metric is a similarity (vs distance)."""
        return self.get(name).is_similarity
    
    def __contains__(self, name: str) -> bool:
        """Check if a metric is registered."""
        canonical = self._aliases.get(name, name)
        return canonical in self._metrics
    
    def __getitem__(self, name: str) -> MetricInfo:
        """Get metric info by name."""
        return self.get(name)


# =============================================================================
# GLOBAL REGISTRY AND CONVENIENCE FUNCTIONS
# =============================================================================

# Global registry instance
_registry = MetricRegistry()


def get_metric(name: str) -> MetricInfo:
    """
    Get metric info by name.
    
    Args:
        name: Metric name or alias
        
    Returns:
        MetricInfo object
        
    Example:
        >>> info = get_metric("euclidean")
        >>> print(info.description)
        'Euclidean (L2) distance'
    """
    return _registry.get(name)


def get_metric_fn(name: str) -> DistanceFunction:
    """
    Get distance function by metric name.
    
    Args:
        name: Metric name or alias
        
    Returns:
        Distance function
        
    Example:
        >>> dist_fn = get_metric_fn("cosine")
        >>> distance = dist_fn(vec_a, vec_b)
    """
    return _registry.get_function(name)


def register_metric(
    name: str,
    function: DistanceFunction,
    is_similarity: bool = False,
    description: str = "",
    batch_function: Optional[BatchDistanceFunction] = None,
    aliases: Optional[List[str]] = None,
) -> None:
    """
    Register a custom distance metric.
    
    Args:
        name: Metric name
        function: Distance function (a, b) -> float
        is_similarity: True if larger values = more similar
        description: Human-readable description
        batch_function: Optional batch distance function
        aliases: Optional list of alternative names
        
    Example:
        >>> def my_distance(a, b):
        ...     return np.sum(a * b)  # Custom metric
        >>> register_metric("my_metric", my_distance, description="My custom metric")
    """
    info = MetricInfo(
        name=name,
        function=function,
        batch_function=batch_function,
        is_similarity=is_similarity,
        min_value=0.0,
        max_value=None,
        description=description or f"Custom metric: {name}",
    )
    _registry.register(info, aliases)


def list_metrics() -> List[str]:
    """
    List all available metric names.
    
    Returns:
        List of metric names
    """
    return _registry.list_metrics()


def is_similarity(name: str) -> bool:
    """
    Check if a metric is a similarity measure.
    
    Args:
        name: Metric name
        
    Returns:
        True if larger values indicate more similarity
    """
    return _registry.is_similarity(name)


def metric_exists(name: str) -> bool:
    """
    Check if a metric is registered.
    
    Args:
        name: Metric name or alias
        
    Returns:
        True if metric exists
    """
    return name in _registry


# =============================================================================
# DISTANCE FUNCTION WRAPPER
# =============================================================================

class DistanceCalculator:
    """
    Wrapper class for distance calculations with a specific metric.
    
    Provides a unified interface for single and batch operations.
    
    Example:
        >>> calc = DistanceCalculator("cosine")
        >>> dist = calc.distance(vec_a, vec_b)
        >>> dists = calc.batch_distances(query, collection)
    """
    
    def __init__(self, metric: str = "euclidean"):
        """
        Initialize calculator with a specific metric.
        
        Args:
            metric: Name of the distance metric
        """
        self.metric = metric
        self.info = get_metric(metric)
        self._fn = self.info.function
        self._batch_fn = self.info.batch_function
    
    def distance(self, a: Vector, b: Vector) -> float:
        """Compute distance between two vectors."""
        return self._fn(a, b)
    
    def batch_distances(
        self, 
        query: Vector, 
        collection: NDArray
    ) -> NDArray:
        """Compute distances from query to all vectors in collection."""
        return query_distances(query, collection, self.metric)
    
    def pairwise(
        self, 
        X: NDArray, 
        Y: Optional[NDArray] = None
    ) -> NDArray:
        """Compute pairwise distances."""
        if self._batch_fn is not None:
            return self._batch_fn(X, Y)
        
        # Fallback to loop-based computation
        if Y is None:
            Y = X
        
        n, m = len(X), len(Y)
        result = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                result[i, j] = self._fn(X[i], Y[j])
        
        return result
    
    @property
    def requires_normalization(self) -> bool:
        """Check if this metric requires normalized vectors."""
        return self.info.requires_normalization
    
    def __repr__(self) -> str:
        return f"DistanceCalculator(metric='{self.metric}')"