"""
Core distance and similarity metric implementations.

All functions are optimized using NumPy vectorized operations.
Distance functions return smaller values for more similar vectors.
"""

from __future__ import annotations

import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray


# Type aliases
Vector = NDArray[np.floating]
VectorBatch = NDArray[np.floating]


# =============================================================================
# SINGLE VECTOR DISTANCE FUNCTIONS
# =============================================================================

def euclidean(a: Vector, b: Vector) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.
    
    Formula: sqrt(sum((a_i - b_i)^2))
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Euclidean distance (>= 0, smaller = more similar)
        
    Example:
        >>> a = np.array([0.0, 0.0])
        >>> b = np.array([3.0, 4.0])
        >>> euclidean(a, b)
        5.0
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))


def euclidean_squared(a: Vector, b: Vector) -> float:
    """
    Compute squared Euclidean distance between two vectors.
    
    Faster than euclidean() as it avoids the sqrt operation.
    Maintains the same ordering as Euclidean distance.
    
    Formula: sum((a_i - b_i)^2)
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Squared Euclidean distance (>= 0, smaller = more similar)
        
    Example:
        >>> a = np.array([0.0, 0.0])
        >>> b = np.array([3.0, 4.0])
        >>> euclidean_squared(a, b)
        25.0
    """
    diff = a - b
    return float(np.dot(diff, diff))


def cosine_similarity(a: Vector, b: Vector, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Formula: (a · b) / (||a|| * ||b||)
    
    Args:
        a: First vector
        b: Second vector
        eps: Small value to avoid division by zero
        
    Returns:
        Cosine similarity in range [-1, 1] (larger = more similar)
        
    Example:
        >>> a = np.array([1.0, 0.0])
        >>> b = np.array([1.0, 0.0])
        >>> cosine_similarity(a, b)
        1.0
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    denominator = norm_a * norm_b
    if denominator < eps:
        return 0.0
    
    return float(dot / denominator)


def cosine_distance(a: Vector, b: Vector, eps: float = 1e-8) -> float:
    """
    Compute cosine distance between two vectors.
    
    Formula: 1 - cosine_similarity(a, b)
    
    Args:
        a: First vector
        b: Second vector
        eps: Small value to avoid division by zero
        
    Returns:
        Cosine distance in range [0, 2] (smaller = more similar)
        
    Example:
        >>> a = np.array([1.0, 0.0])
        >>> b = np.array([0.0, 1.0])
        >>> cosine_distance(a, b)
        1.0
    """
    return 1.0 - cosine_similarity(a, b, eps)


def dot_product(a: Vector, b: Vector) -> float:
    """
    Compute dot product (inner product) between two vectors.
    
    Formula: sum(a_i * b_i)
    
    For normalized vectors, this equals cosine similarity.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Dot product (larger = more similar for normalized vectors)
        
    Example:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([4.0, 5.0, 6.0])
        >>> dot_product(a, b)
        32.0
    """
    return float(np.dot(a, b))


def negative_dot_product(a: Vector, b: Vector) -> float:
    """
    Compute negative dot product (for use as distance).
    
    Formula: -sum(a_i * b_i)
    
    This converts dot product to a distance metric where
    smaller values indicate more similar vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Negative dot product (smaller = more similar)
    """
    return -float(np.dot(a, b))


def manhattan(a: Vector, b: Vector) -> float:
    """
    Compute Manhattan (L1) distance between two vectors.
    
    Also known as taxicab distance or city block distance.
    
    Formula: sum(|a_i - b_i|)
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Manhattan distance (>= 0, smaller = more similar)
        
    Example:
        >>> a = np.array([0.0, 0.0])
        >>> b = np.array([3.0, 4.0])
        >>> manhattan(a, b)
        7.0
    """
    return float(np.sum(np.abs(a - b)))


def chebyshev(a: Vector, b: Vector) -> float:
    """
    Compute Chebyshev (L∞) distance between two vectors.
    
    Also known as chessboard distance.
    
    Formula: max(|a_i - b_i|)
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Chebyshev distance (>= 0, smaller = more similar)
        
    Example:
        >>> a = np.array([0.0, 0.0])
        >>> b = np.array([3.0, 4.0])
        >>> chebyshev(a, b)
        4.0
    """
    return float(np.max(np.abs(a - b)))


def hamming(a: Vector, b: Vector) -> float:
    """
    Compute Hamming distance between two vectors.
    
    Counts the number of positions where elements differ.
    Typically used for binary vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Hamming distance (number of differing elements)
        
    Example:
        >>> a = np.array([1, 0, 1, 1, 0])
        >>> b = np.array([1, 1, 1, 0, 0])
        >>> hamming(a, b)
        2.0
    """
    return float(np.sum(a != b))


def minkowski(a: Vector, b: Vector, p: float = 2.0) -> float:
    """
    Compute Minkowski distance between two vectors.
    
    Generalization of Euclidean (p=2) and Manhattan (p=1) distances.
    
    Formula: (sum(|a_i - b_i|^p))^(1/p)
    
    Args:
        a: First vector
        b: Second vector
        p: Order of the norm (p >= 1)
        
    Returns:
        Minkowski distance (>= 0, smaller = more similar)
        
    Example:
        >>> a = np.array([0.0, 0.0])
        >>> b = np.array([3.0, 4.0])
        >>> minkowski(a, b, p=2)  # Same as Euclidean
        5.0
    """
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    
    return float(np.power(np.sum(np.power(np.abs(a - b), p)), 1.0 / p))


# =============================================================================
# PAIRWISE DISTANCE FUNCTIONS (BATCH OPERATIONS)
# =============================================================================

def pairwise_euclidean(X: VectorBatch, Y: Optional[VectorBatch] = None) -> NDArray:
    """
    Compute pairwise Euclidean distances between vectors.
    
    Uses the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d), or None to compute X vs X
        
    Returns:
        Distance matrix of shape (n, m) or (n, n)
        
    Example:
        >>> X = np.array([[0, 0], [1, 1]])
        >>> pairwise_euclidean(X)
        array([[0.        , 1.41421356],
               [1.41421356, 0.        ]])
    """
    if Y is None:
        Y = X
    
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_sqnorm = np.sum(Y ** 2, axis=1, keepdims=True)  # (m, 1)
    
    # Dot product matrix
    dot_product_matrix = X @ Y.T  # (n, m)
    
    # Squared distances
    sq_distances = X_sqnorm + Y_sqnorm.T - 2 * dot_product_matrix
    
    # Clamp negative values (numerical errors) and sqrt
    sq_distances = np.maximum(sq_distances, 0)
    return np.sqrt(sq_distances)


def pairwise_euclidean_squared(
    X: VectorBatch, 
    Y: Optional[VectorBatch] = None
) -> NDArray:
    """
    Compute pairwise squared Euclidean distances.
    
    Faster than pairwise_euclidean() as it avoids sqrt.
    
    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d), or None to compute X vs X
        
    Returns:
        Squared distance matrix of shape (n, m) or (n, n)
    """
    if Y is None:
        Y = X
    
    X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sqnorm = np.sum(Y ** 2, axis=1, keepdims=True)
    dot_product_matrix = X @ Y.T
    
    sq_distances = X_sqnorm + Y_sqnorm.T - 2 * dot_product_matrix
    return np.maximum(sq_distances, 0)


def pairwise_cosine(
    X: VectorBatch, 
    Y: Optional[VectorBatch] = None,
    eps: float = 1e-8
) -> NDArray:
    """
    Compute pairwise cosine distances between vectors.
    
    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d), or None to compute X vs X
        eps: Small value to avoid division by zero
        
    Returns:
        Distance matrix of shape (n, m) or (n, n)
        Values in range [0, 2]
    """
    if Y is None:
        Y = X
    
    # Normalize vectors
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    
    X_norm = np.where(X_norm < eps, 1.0, X_norm)
    Y_norm = np.where(Y_norm < eps, 1.0, Y_norm)
    
    X_normalized = X / X_norm
    Y_normalized = Y / Y_norm
    
    # Cosine similarity
    similarity = X_normalized @ Y_normalized.T
    
    # Clamp to [-1, 1] for numerical stability
    similarity = np.clip(similarity, -1.0, 1.0)
    
    # Convert to distance
    return 1.0 - similarity


def pairwise_dot(X: VectorBatch, Y: Optional[VectorBatch] = None) -> NDArray:
    """
    Compute pairwise negative dot products (as distances).
    
    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d), or None to compute X vs X
        
    Returns:
        Negative dot product matrix of shape (n, m) or (n, n)
        Smaller values = more similar
    """
    if Y is None:
        Y = X
    
    return -(X @ Y.T)


def pairwise_manhattan(X: VectorBatch, Y: Optional[VectorBatch] = None) -> NDArray:
    """
    Compute pairwise Manhattan distances.
    
    Note: This is O(n*m*d) and memory-intensive for large batches.
    
    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d), or None to compute X vs X
        
    Returns:
        Distance matrix of shape (n, m) or (n, n)
    """
    if Y is None:
        Y = X
    
    n = X.shape[0]
    m = Y.shape[0]
    
    # For small batches, use broadcasting
    if n * m < 10000:
        # X[:, np.newaxis, :] has shape (n, 1, d)
        # Y[np.newaxis, :, :] has shape (1, m, d)
        return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=2)
    
    # For larger batches, compute row by row to save memory
    result = np.zeros((n, m), dtype=X.dtype)
    for i in range(n):
        result[i] = np.sum(np.abs(X[i] - Y), axis=1)
    
    return result


# =============================================================================
# QUERY-TO-COLLECTION DISTANCES
# =============================================================================

def query_distances(
    query: Vector,
    collection: VectorBatch,
    metric: str = "euclidean",
) -> NDArray:
    """
    Compute distances from a query vector to all vectors in a collection.
    
    Args:
        query: Query vector of shape (d,)
        collection: Collection of vectors of shape (n, d)
        metric: Distance metric to use
        
    Returns:
        Distances array of shape (n,)
        
    Example:
        >>> query = np.array([0.0, 0.0])
        >>> collection = np.array([[1, 0], [0, 1], [1, 1]])
        >>> query_distances(query, collection, "euclidean")
        array([1.        , 1.        , 1.41421356])
    """
    if metric == "euclidean":
        # ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
        q_sqnorm = np.dot(query, query)
        c_sqnorm = np.sum(collection ** 2, axis=1)
        dots = collection @ query
        sq_distances = q_sqnorm + c_sqnorm - 2 * dots
        return np.sqrt(np.maximum(sq_distances, 0))
    
    elif metric == "euclidean_squared":
        q_sqnorm = np.dot(query, query)
        c_sqnorm = np.sum(collection ** 2, axis=1)
        dots = collection @ query
        sq_distances = q_sqnorm + c_sqnorm - 2 * dots
        return np.maximum(sq_distances, 0)
    
    elif metric == "cosine":
        q_norm = np.linalg.norm(query)
        c_norms = np.linalg.norm(collection, axis=1)
        
        if q_norm < 1e-8:
            return np.ones(len(collection))
        
        # Avoid division by zero
        c_norms = np.where(c_norms < 1e-8, 1.0, c_norms)
        
        dots = collection @ query
        similarities = dots / (q_norm * c_norms)
        similarities = np.clip(similarities, -1.0, 1.0)
        return 1.0 - similarities
    
    elif metric == "dot":
        return -(collection @ query)
    
    elif metric == "manhattan":
        return np.sum(np.abs(collection - query), axis=1)
    
    elif metric == "chebyshev":
        return np.max(np.abs(collection - query), axis=1)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =============================================================================
# OPTIMIZED SINGLE VECTOR OPERATIONS (for normalized vectors)
# =============================================================================

def euclidean_normalized(a: Vector, b: Vector) -> float:
    """
    Compute Euclidean distance between normalized vectors.
    
    For unit vectors: ||a-b||^2 = 2 - 2*a·b = 2*(1 - cos(θ))
    
    This is faster as it only needs the dot product.
    
    Args:
        a: First normalized vector
        b: Second normalized vector
        
    Returns:
        Euclidean distance
    """
    dot = np.dot(a, b)
    # Clamp dot product to [-1, 1] for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.sqrt(2.0 - 2.0 * dot))


def angular_distance(a: Vector, b: Vector, eps: float = 1e-8) -> float:
    """
    Compute angular distance (angle in radians) between vectors.
    
    Formula: arccos(cosine_similarity(a, b))
    
    Args:
        a: First vector
        b: Second vector
        eps: Small value to avoid division by zero
        
    Returns:
        Angular distance in radians [0, π]
    """
    cos_sim = cosine_similarity(a, b, eps)
    # Clamp for numerical stability
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.arccos(cos_sim))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_vectors(vectors: VectorBatch, eps: float = 1e-8) -> VectorBatch:
    """
    Normalize vectors to unit length.
    
    Args:
        vectors: Array of shape (n, d)
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized vectors of shape (n, d)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return vectors / norms


def compute_centroid(vectors: VectorBatch) -> Vector:
    """
    Compute the centroid (mean) of a set of vectors.
    
    Args:
        vectors: Array of shape (n, d)
        
    Returns:
        Centroid vector of shape (d,)
    """
    return np.mean(vectors, axis=0)


def compute_medoid(
    vectors: VectorBatch, 
    metric: str = "euclidean"
) -> tuple[int, Vector]:
    """
    Find the medoid (most central point) of a set of vectors.
    
    The medoid is the point that minimizes the sum of distances
    to all other points.
    
    Args:
        vectors: Array of shape (n, d)
        metric: Distance metric to use
        
    Returns:
        Tuple of (medoid_index, medoid_vector)
    """
    distances = query_distances_matrix(vectors, metric)
    sum_distances = np.sum(distances, axis=1)
    medoid_idx = int(np.argmin(sum_distances))
    return medoid_idx, vectors[medoid_idx]


def query_distances_matrix(
    vectors: VectorBatch,
    metric: str = "euclidean"
) -> NDArray:
    """
    Compute pairwise distance matrix.
    
    Args:
        vectors: Array of shape (n, d)
        metric: Distance metric
        
    Returns:
        Distance matrix of shape (n, n)
    """
    if metric == "euclidean":
        return pairwise_euclidean(vectors)
    elif metric == "euclidean_squared":
        return pairwise_euclidean_squared(vectors)
    elif metric == "cosine":
        return pairwise_cosine(vectors)
    elif metric == "dot":
        return pairwise_dot(vectors)
    elif metric == "manhattan":
        return pairwise_manhattan(vectors)
    else:
        raise ValueError(f"Unknown metric: {metric}")