"""
Unit tests for VectorRecord and VectorBatch.
"""

import pytest
import numpy as np
from vectordb.core.vector import VectorRecord, VectorBatch, validate_vector


class TestVectorRecord:
    """Tests for VectorRecord class."""
    
    def test_create_basic(self, dimension):
        """Test basic record creation."""
        vector = np.random.randn(dimension).astype(np.float32)
        record = VectorRecord(id="test", vector=vector)
        
        assert record.id == "test"
        assert record.dimension == dimension
        assert np.array_equal(record.vector, vector)
        assert record.metadata == {}
    
    def test_create_with_metadata(self, random_vector):
        """Test record creation with metadata."""
        metadata = {"key": "value", "number": 42}
        record = VectorRecord(
            id="test",
            vector=random_vector,
            metadata=metadata,
        )
        
        assert record.metadata == metadata
    
    def test_create_from_list(self, dimension):
        """Test creating record from Python list."""
        vector_list = [0.1] * dimension
        record = VectorRecord(id="test", vector=vector_list)
        
        assert isinstance(record.vector, np.ndarray)
        assert record.vector.dtype == np.float32
    
    def test_auto_convert_to_float32(self, dimension):
        """Test automatic conversion to float32."""
        vector = np.random.randn(dimension).astype(np.float64)
        record = VectorRecord(id="test", vector=vector)
        
        assert record.vector.dtype == np.float32
    
    def test_dimension_property(self, random_vector):
        """Test dimension property."""
        record = VectorRecord(id="test", vector=random_vector)
        assert record.dimension == len(random_vector)
    
    def test_norm_property(self):
        """Test norm property."""
        vector = np.array([3.0, 4.0], dtype=np.float32)
        record = VectorRecord(id="test", vector=vector)
        
        assert record.norm == pytest.approx(5.0)
    
    def test_normalize(self):
        """Test vector normalization."""
        vector = np.array([3.0, 4.0], dtype=np.float32)
        record = VectorRecord(id="test", vector=vector)
        normalized = record.normalize()
        
        assert normalized.norm == pytest.approx(1.0)
        assert np.allclose(normalized.vector, [0.6, 0.8])
    
    def test_normalize_preserves_original(self, random_vector):
        """Test that normalize() doesn't modify original."""
        record = VectorRecord(id="test", vector=random_vector)
        original_vector = record.vector.copy()
        
        normalized = record.normalize()
        
        assert np.array_equal(record.vector, original_vector)
    
    def test_copy(self, sample_record):
        """Test deep copy."""
        copy = sample_record.copy()
        
        assert copy.id == sample_record.id
        assert np.array_equal(copy.vector, sample_record.vector)
        assert copy.metadata == sample_record.metadata
        
        # Verify it's a deep copy
        copy.metadata["new_key"] = "new_value"
        assert "new_key" not in sample_record.metadata
    
    def test_update_metadata(self, sample_record):
        """Test metadata update."""
        updated = sample_record.update_metadata({"new_key": "new_value"})
        
        assert updated.id == sample_record.id
        assert "new_key" in updated.metadata
        assert "category" in updated.metadata  # Original key preserved
    
    def test_to_dict_and_back(self, sample_record):
        """Test serialization round-trip."""
        data = sample_record.to_dict()
        restored = VectorRecord.from_dict(data)
        
        assert restored.id == sample_record.id
        assert np.allclose(restored.vector, sample_record.vector)
        assert restored.metadata == sample_record.metadata
    
    def test_equality(self, random_vector):
        """Test equality comparison."""
        record1 = VectorRecord(id="test", vector=random_vector.copy())
        record2 = VectorRecord(id="test", vector=random_vector.copy())
        
        assert record1 == record2
    
    def test_inequality_different_id(self, random_vector):
        """Test inequality with different ID."""
        record1 = VectorRecord(id="test1", vector=random_vector)
        record2 = VectorRecord(id="test2", vector=random_vector)
        
        assert record1 != record2
    
    def test_invalid_empty_vector(self):
        """Test rejection of empty vector."""
        with pytest.raises(ValueError, match="empty"):
            VectorRecord(id="test", vector=np.array([]))
    
    def test_invalid_nan_vector(self, dimension):
        """Test rejection of NaN values."""
        vector = np.array([np.nan] + [0.0] * (dimension - 1))
        with pytest.raises(ValueError, match="NaN"):
            VectorRecord(id="test", vector=vector)
    
    def test_invalid_inf_vector(self, dimension):
        """Test rejection of Inf values."""
        vector = np.array([np.inf] + [0.0] * (dimension - 1))
        with pytest.raises(ValueError, match="Inf"):
            VectorRecord(id="test", vector=vector)
    
    def test_invalid_2d_vector(self):
        """Test rejection of 2D vector."""
        vector = np.array([[1, 2], [3, 4]], dtype=np.float32)
        with pytest.raises(ValueError, match="1-dimensional"):
            VectorRecord(id="test", vector=vector)


class TestVectorBatch:
    """Tests for VectorBatch class."""
    
    def test_create_empty(self, dimension):
        """Test creating empty batch."""
        batch = VectorBatch(dimension=dimension)
        
        assert len(batch) == 0
        assert batch.dimension == dimension
    
    def test_add_vector(self, dimension):
        """Test adding vectors to batch."""
        batch = VectorBatch(dimension=dimension)
        
        for i in range(10):
            vector = np.random.randn(dimension).astype(np.float32)
            batch.add(f"id_{i}", vector, {"index": i})
        
        assert len(batch) == 10
    
    def test_vectors_property(self, sample_batch):
        """Test vectors property returns 2D array."""
        vectors = sample_batch.vectors
        
        assert isinstance(vectors, np.ndarray)
        assert vectors.ndim == 2
        assert vectors.shape[0] == len(sample_batch)
    
    def test_from_records(self, sample_records):
        """Test creating batch from records."""
        batch = VectorBatch.from_records(sample_records)
        
        assert len(batch) == len(sample_records)
    
    def test_from_numpy(self, random_vectors):
        """Test creating batch from numpy array."""
        batch = VectorBatch.from_numpy(random_vectors)
        
        assert len(batch) == len(random_vectors)
        assert np.allclose(batch.vectors, random_vectors)
    
    def test_iteration(self, sample_batch):
        """Test iterating over batch."""
        count = 0
        for record in sample_batch:
            assert isinstance(record, VectorRecord)
            count += 1
        
        assert count == len(sample_batch)
    
    def test_indexing(self, sample_batch):
        """Test getting record by index."""
        record = sample_batch[0]
        
        assert isinstance(record, VectorRecord)
        assert record.id == sample_batch.ids[0]
    
    def test_get_by_id(self, sample_batch):
        """Test getting record by ID."""
        expected_id = sample_batch.ids[5]
        record = sample_batch.get_by_id(expected_id)
        
        assert record is not None
        assert record.id == expected_id
    
    def test_get_by_id_not_found(self, sample_batch):
        """Test getting non-existent ID."""
        record = sample_batch.get_by_id("nonexistent")
        assert record is None
    
    def test_normalize_all(self, sample_batch):
        """Test normalizing all vectors."""
        normalized = sample_batch.normalize_all()
        
        norms = np.linalg.norm(normalized.vectors, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_auto_expand(self, dimension):
        """Test automatic capacity expansion."""
        batch = VectorBatch(dimension=dimension, initial_capacity=10)
        
        # Add more than initial capacity
        for i in range(100):
            vector = np.random.randn(dimension).astype(np.float32)
            batch.add(f"id_{i}", vector)
        
        assert len(batch) == 100
    
    def test_dimension_mismatch(self, dimension):
        """Test rejection of wrong dimension."""
        batch = VectorBatch(dimension=dimension)
        wrong_vector = np.random.randn(dimension + 10).astype(np.float32)
        
        with pytest.raises(ValueError, match="dimension"):
            batch.add("id", wrong_vector)


class TestValidateVector:
    """Tests for validate_vector function."""
    
    def test_validate_list(self, dimension):
        """Test validating Python list."""
        vector_list = [0.1] * dimension
        result = validate_vector(vector_list)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
    
    def test_validate_with_expected_dimension(self, dimension):
        """Test validation with expected dimension."""
        vector = np.random.randn(dimension).astype(np.float32)
        result = validate_vector(vector, expected_dimension=dimension)
        
        assert len(result) == dimension
    
    def test_validate_wrong_dimension(self, dimension):
        """Test rejection of wrong dimension."""
        vector = np.random.randn(dimension).astype(np.float32)
        
        with pytest.raises(ValueError, match="dimension"):
            validate_vector(vector, expected_dimension=dimension + 1)
    
    def test_validate_with_normalization(self, random_vector):
        """Test validation with normalization."""
        result = validate_vector(random_vector, normalize=True)
        
        assert np.linalg.norm(result) == pytest.approx(1.0)
    
    def test_validate_empty_vector(self):
        """Test rejection of empty vector."""
        with pytest.raises(ValueError, match="empty"):
            validate_vector([])
    
    def test_validate_2d_array(self):
        """Test rejection of 2D array."""
        with pytest.raises(ValueError, match="1D"):
            validate_vector(np.array([[1, 2], [3, 4]]))