"""
Batch operation utilities for VectorDB.

Provides efficient batching for large-scale operations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Iterator, Tuple, Callable, Optional, TypeVar
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

T = TypeVar('T')


@dataclass
class BatchConfig:
    """Configuration for batch operations."""
    batch_size: int = 1000
    max_workers: int = 4
    show_progress: bool = False


class BatchIterator:
    """
    Iterator that yields batches from a large dataset.
    
    Example:
        >>> data = list(range(10000))
        >>> for batch in BatchIterator(data, batch_size=100):
        ...     process(batch)
    """
    
    def __init__(
        self,
        data: List[T],
        batch_size: int = 1000,
    ):
        self.data = data
        self.batch_size = batch_size
        self._index = 0
    
    def __iter__(self) -> 'BatchIterator':
        self._index = 0
        return self
    
    def __next__(self) -> List[T]:
        if self._index >= len(self.data):
            raise StopIteration
        
        batch = self.data[self._index:self._index + self.batch_size]
        self._index += self.batch_size
        return batch
    
    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class VectorBatchIterator:
    """
    Iterator for batching vectors with their IDs and metadata.
    
    Example:
        >>> iterator = VectorBatchIterator(ids, vectors, metadata, batch_size=1000)
        >>> for batch_ids, batch_vectors, batch_metadata in iterator:
        ...     index.add_batch(batch_ids, batch_vectors, batch_metadata)
    """
    
    def __init__(
        self,
        ids: List[str],
        vectors: NDArray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 1000,
    ):
        self.ids = ids
        self.vectors = vectors
        self.metadata = metadata or [{} for _ in ids]
        self.batch_size = batch_size
        self._index = 0
    
    def __iter__(self) -> 'VectorBatchIterator':
        self._index = 0
        return self
    
    def __next__(self) -> Tuple[List[str], NDArray, List[Dict[str, Any]]]:
        if self._index >= len(self.ids):
            raise StopIteration
        
        end = min(self._index + self.batch_size, len(self.ids))
        
        batch_ids = self.ids[self._index:end]
        batch_vectors = self.vectors[self._index:end]
        batch_metadata = self.metadata[self._index:end]
        
        self._index = end
        return batch_ids, batch_vectors, batch_metadata
    
    def __len__(self) -> int:
        return (len(self.ids) + self.batch_size - 1) // self.batch_size


def batch_process(
    items: List[T],
    process_fn: Callable[[List[T]], Any],
    batch_size: int = 1000,
    show_progress: bool = False,
) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: Items to process
        process_fn: Function to apply to each batch
        batch_size: Size of each batch
        show_progress: Print progress
        
    Returns:
        List of results from each batch
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i, batch in enumerate(BatchIterator(items, batch_size)):
        result = process_fn(batch)
        results.append(result)
        
        if show_progress:
            print(f"Batch {i + 1}/{total_batches} completed")
    
    return results


def parallel_batch_process(
    items: List[T],
    process_fn: Callable[[List[T]], Any],
    batch_size: int = 1000,
    max_workers: int = 4,
    show_progress: bool = False,
) -> List[Any]:
    """
    Process items in batches using multiple threads.
    
    Args:
        items: Items to process
        process_fn: Function to apply to each batch
        batch_size: Size of each batch
        max_workers: Number of parallel workers
        show_progress: Print progress
        
    Returns:
        List of results from each batch
    """
    batches = list(BatchIterator(items, batch_size))
    results = [None] * len(batches)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_fn, batch): i
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            completed += 1
            
            if show_progress:
                print(f"Batch {completed}/{len(batches)} completed")
    
    return results


class BatchAccumulator:
    """
    Accumulates items and flushes when batch size is reached.
    
    Example:
        >>> accumulator = BatchAccumulator(batch_size=100, flush_fn=process_batch)
        >>> for item in items:
        ...     accumulator.add(item)
        >>> accumulator.flush()  # Process remaining items
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        flush_fn: Optional[Callable[[List[Any]], None]] = None,
    ):
        self.batch_size = batch_size
        self.flush_fn = flush_fn
        self._buffer: List[Any] = []
        self._lock = threading.Lock()
    
    def add(self, item: Any) -> bool:
        """
        Add an item to the accumulator.
        
        Returns True if a flush occurred.
        """
        with self._lock:
            self._buffer.append(item)
            
            if len(self._buffer) >= self.batch_size:
                self._do_flush()
                return True
            
            return False
    
    def add_many(self, items: List[Any]) -> int:
        """
        Add multiple items.
        
        Returns number of flushes that occurred.
        """
        flushes = 0
        for item in items:
            if self.add(item):
                flushes += 1
        return flushes
    
    def flush(self) -> None:
        """Flush any remaining items."""
        with self._lock:
            if self._buffer:
                self._do_flush()
    
    def _do_flush(self) -> None:
        """Internal flush."""
        if self.flush_fn:
            self.flush_fn(self._buffer)
        self._buffer = []
    
    @property
    def pending(self) -> int:
        """Number of pending items."""
        return len(self._buffer)
    
    def __enter__(self) -> 'BatchAccumulator':
        return self
    
    def __exit__(self, *args) -> None:
        self.flush()


class RateLimitedBatcher:
    """
    Batcher with rate limiting.
    
    Useful for API calls or resource-constrained operations.
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_rate: float = 10.0,  # batches per second
    ):
        self.batch_size = batch_size
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate
        self._last_batch_time = 0.0
    
    def process(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], Any],
    ) -> List[Any]:
        """Process items with rate limiting."""
        results = []
        
        for batch in BatchIterator(items, self.batch_size):
            # Rate limit
            elapsed = time.time() - self._last_batch_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            
            result = process_fn(batch)
            results.append(result)
            
            self._last_batch_time = time.time()
        
        return results