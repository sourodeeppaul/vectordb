"""
Pytest fixtures for VectorDB tests.
"""

import pytest
import numpy as np
from typing import List

from vectordb.core.vector import VectorRecord, VectorBatch


@pytest.fixture
def dimension() -> int:
    """Default dimension for test vectors."""
    return 128


@pytest.fixture
def random_vector(dimension: int) -> np.ndarray:
    """Generate a random vector."""
    return np.random.randn(dimension).astype(np.float32)


@pytest.fixture
def random_vectors(dimension: int) -> np.ndarray:
    """Generate random vectors (100 vectors)."""
    return np.random.randn(100, dimension).astype(np.float32)


@pytest.fixture
def sample_record(random_vector: np.ndarray) -> VectorRecord:
    """Create a sample VectorRecord."""
    return VectorRecord(
        id="test_001",
        vector=random_vector,
        metadata={"category": "test", "value": 42},
    )


@pytest.fixture
def sample_records(dimension: int) -> List[VectorRecord]:
    """Create sample VectorRecords."""
    records = []
    for i in range(100):
        records.append(
            VectorRecord(
                id=f"record_{i:03d}",
                vector=np.random.randn(dimension).astype(np.float32),
                metadata={"index": i, "category": f"cat_{i % 5}"},
            )
        )
    return records


@pytest.fixture
def sample_batch(sample_records: List[VectorRecord]) -> VectorBatch:
    """Create a sample VectorBatch."""
    return VectorBatch.from_records(sample_records)


@pytest.fixture
def normalized_vectors(random_vectors: np.ndarray) -> np.ndarray:
    """Generate normalized random vectors."""
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    return random_vectors / norms