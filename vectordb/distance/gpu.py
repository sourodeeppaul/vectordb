"""
GPU-accelerated distance calculations.

This module provides GPU-accelerated distance functions using CuPy
when available, with automatic fallback to CPU implementations.

Note:
    GPU acceleration requires CuPy and a CUDA-capable GPU.
    Install with: pip install cupy-cuda11x (adjust for your CUDA version)
"""

import numpy as np
from typing import Optional, Union, Callable

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


class GPUDistanceCalculator:
    """
    GPU-accelerated distance calculator.
    
    Automatically uses GPU when available and data is large enough
    to benefit from GPU acceleration.
    
    Attributes:
        metric: Distance metric name
        min_vectors_for_gpu: Minimum vectors to use GPU
        device_id: CUDA device ID
        
    Example:
        >>> calc = GPUDistanceCalculator("euclidean")
        >>> distances = calc.compute(query, vectors)
    """
    
    # Minimum vectors to warrant GPU transfer overhead
    MIN_VECTORS_FOR_GPU = 10000
    
    def __init__(
        self,
        metric: str = "euclidean",
        min_vectors_for_gpu: int = 10000,
        device_id: int = 0,
    ):
        """
        Initialize GPU distance calculator.
        
        Args:
            metric: Distance metric (euclidean, cosine, dot)
            min_vectors_for_gpu: Minimum vectors to use GPU
            device_id: CUDA device ID to use
        """
        self.metric = metric
        self.min_vectors_for_gpu = min_vectors_for_gpu
        self.device_id = device_id
        self._gpu_available = HAS_GPU
        
        if self._gpu_available:
            try:
                cp.cuda.Device(device_id).use()
            except Exception:
                self._gpu_available = False
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self._gpu_available
    
    def compute(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        force_gpu: bool = False,
        force_cpu: bool = False,
    ) -> np.ndarray:
        """
        Compute distances from query to all vectors.
        
        Automatically chooses GPU or CPU based on data size
        and availability.
        
        Args:
            query: Query vector of shape (d,)
            vectors: Matrix of vectors of shape (n, d)
            force_gpu: Force GPU usage (raises if unavailable)
            force_cpu: Force CPU usage
            
        Returns:
            Array of distances of shape (n,)
        """
        n = vectors.shape[0]
        use_gpu = (
            self._gpu_available
            and not force_cpu
            and (force_gpu or n >= self.min_vectors_for_gpu)
        )
        
        if force_gpu and not self._gpu_available:
            raise RuntimeError("GPU requested but CuPy not available")
        
        if use_gpu:
            return self._compute_gpu(query, vectors)
        else:
            return self._compute_cpu(query, vectors)
    
    def _compute_gpu(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute distances on GPU."""
        # Transfer to GPU
        query_gpu = cp.asarray(query, dtype=cp.float32)
        vectors_gpu = cp.asarray(vectors, dtype=cp.float32)
        
        # Compute distances
        if self.metric == "euclidean":
            diff = vectors_gpu - query_gpu
            distances_gpu = cp.sqrt(cp.sum(diff * diff, axis=1))
        elif self.metric == "cosine":
            query_norm = cp.linalg.norm(query_gpu)
            vectors_norm = cp.linalg.norm(vectors_gpu, axis=1)
            similarity = cp.dot(vectors_gpu, query_gpu) / (query_norm * vectors_norm + 1e-10)
            distances_gpu = 1.0 - similarity
        elif self.metric == "dot":
            distances_gpu = -cp.dot(vectors_gpu, query_gpu)
        elif self.metric == "manhattan":
            distances_gpu = cp.sum(cp.abs(vectors_gpu - query_gpu), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Transfer back to CPU
        return cp.asnumpy(distances_gpu)
    
    def _compute_cpu(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute distances on CPU."""
        if self.metric == "euclidean":
            diff = vectors - query
            return np.sqrt(np.sum(diff * diff, axis=1))
        elif self.metric == "cosine":
            query_norm = np.linalg.norm(query)
            vectors_norm = np.linalg.norm(vectors, axis=1)
            similarity = np.dot(vectors, query) / (query_norm * vectors_norm + 1e-10)
            return 1.0 - similarity
        elif self.metric == "dot":
            return -np.dot(vectors, query)
        elif self.metric == "manhattan":
            return np.sum(np.abs(vectors - query), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


def gpu_euclidean_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated Euclidean distance calculation.
    
    Falls back to CPU if GPU unavailable.
    """
    if not HAS_GPU:
        diff = vectors - query
        return np.sqrt(np.sum(diff * diff, axis=1))
    
    query_gpu = cp.asarray(query, dtype=cp.float32)
    vectors_gpu = cp.asarray(vectors, dtype=cp.float32)
    diff = vectors_gpu - query_gpu
    distances = cp.sqrt(cp.sum(diff * diff, axis=1))
    return cp.asnumpy(distances)


def gpu_cosine_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated cosine distance calculation.
    
    Falls back to CPU if GPU unavailable.
    """
    if not HAS_GPU:
        query_norm = np.linalg.norm(query)
        vectors_norm = np.linalg.norm(vectors, axis=1)
        similarity = np.dot(vectors, query) / (query_norm * vectors_norm + 1e-10)
        return 1.0 - similarity
    
    query_gpu = cp.asarray(query, dtype=cp.float32)
    vectors_gpu = cp.asarray(vectors, dtype=cp.float32)
    query_norm = cp.linalg.norm(query_gpu)
    vectors_norm = cp.linalg.norm(vectors_gpu, axis=1)
    similarity = cp.dot(vectors_gpu, query_gpu) / (query_norm * vectors_norm + 1e-10)
    return cp.asnumpy(1.0 - similarity)


def gpu_dot_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated dot product calculation.
    
    Falls back to CPU if GPU unavailable.
    """
    if not HAS_GPU:
        return np.dot(vectors, query)
    
    query_gpu = cp.asarray(query, dtype=cp.float32)
    vectors_gpu = cp.asarray(vectors, dtype=cp.float32)
    return cp.asnumpy(cp.dot(vectors_gpu, query_gpu))


def get_gpu_info() -> dict:
    """
    Get GPU information.
    
    Returns:
        Dictionary with GPU info or empty dict if unavailable.
    """
    if not HAS_GPU:
        return {"available": False}
    
    try:
        device = cp.cuda.Device()
        return {
            "available": True,
            "device_id": device.id,
            "name": cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode(),
            "memory_total": device.mem_info[1],
            "memory_free": device.mem_info[0],
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


__all__ = [
    "HAS_GPU",
    "GPUDistanceCalculator",
    "gpu_euclidean_batch",
    "gpu_cosine_batch",
    "gpu_dot_batch",
    "get_gpu_info",
]
