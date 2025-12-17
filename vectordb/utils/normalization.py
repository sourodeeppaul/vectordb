"""
Vector normalization utilities.
"""

import numpy as np
from typing import Optional


def normalize_vector(
    vector: np.ndarray,
    copy: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize a vector to unit length (L2 normalization).
    
    Args:
        vector: Input vector
        copy: Whether to create a copy (False modifies in-place)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Normalized vector
    """
    if copy:
        vector = vector.copy()
    
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector  # Return as-is for zero/near-zero vectors
    
    vector /= norm
    return vector


def normalize_batch(
    vectors: np.ndarray,
    copy: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize a batch of vectors to unit length.
    
    Args:
        vectors: Input vectors of shape (n, dim)
        copy: Whether to create a copy
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Normalized vectors of shape (n, dim)
    """
    if copy:
        vectors = vectors.copy()
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)  # Avoid division by zero
    
    vectors /= norms
    return vectors


def compute_norms(vectors: np.ndarray) -> np.ndarray:
    """
    Compute L2 norms for a batch of vectors.
    
    Args:
        vectors: Input vectors of shape (n, dim)
        
    Returns:
        Norms of shape (n,)
    """
    return np.linalg.norm(vectors, axis=1)


def is_normalized(
    vector: np.ndarray,
    tolerance: float = 1e-6,
) -> bool:
    """
    Check if a vector is normalized (unit length).
    
    Args:
        vector: Input vector
        tolerance: Tolerance for comparison
        
    Returns:
        True if vector is normalized
    """
    norm = np.linalg.norm(vector)
    return abs(norm - 1.0) < tolerance


def batch_is_normalized(
    vectors: np.ndarray,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Check if vectors in a batch are normalized.
    
    Args:
        vectors: Input vectors of shape (n, dim)
        tolerance: Tolerance for comparison
        
    Returns:
        Boolean array of shape (n,)
    """
    norms = np.linalg.norm(vectors, axis=1)
    return np.abs(norms - 1.0) < tolerance