"""
Unit tests for VectorDB class.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from vectordb import VectorDB
from vectordb.core.exceptions import (
    CollectionNotFoundError,
    CollectionExistsError,
)


class TestVectorDBBasics:
    """Basic database tests."""
    
    def test_create_in_memory(self):
        """Test creating in-memory database."""
        db = VectorDB()
        
        assert len(db) == 0
        assert db.info()["data_dir"] is None
    
    def test_create_with_data_dir(self):
        """Test creating with data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VectorDB(data_dir=tmpdir)
            
            assert db.info()["data_dir"] == tmpdir
    
    def test_context_manager(self):
        """Test using as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VectorDB(data_dir=tmpdir) as db:
                db.create_collection("test", dimension=10)
                assert len(db) == 1


class TestCollectionManagement:
    """Collection management tests."""
    
    @pytest.fixture
    def db(self):
        return VectorDB()
    
    def test_create_collection(self, db):
        """Test creating a collection."""
        collection = db.create_collection("test", dimension=128)
        
        assert collection.name == "test"
        assert collection.dimension == 128
        assert "test" in db
    
    def test_create_duplicate_raises(self, db):
        """Test duplicate collection rejection."""
        db.create_collection("test", dimension=128)
        
        with pytest.raises(CollectionExistsError):
            db.create_collection("test", dimension=128)
    
    def test_create_exist_ok(self, db):
        """Test exist_ok mode."""
        c1 = db.create_collection("test", dimension=128)
        c2 = db.create_collection("test", dimension=128, exist_ok=True)
        
        assert c1 is c2
    
    def test_get_collection(self, db):
        """Test getting a collection."""
        db.create_collection("test", dimension=128)
        
        collection = db.get_collection("test")
        
        assert collection.name == "test"
    
    def test_get_not_found(self, db):
        """Test getting non-existent collection."""
        with pytest.raises(CollectionNotFoundError):
            db.get_collection("nonexistent")
    
    def test_get_or_create(self, db):
        """Test get_or_create_collection."""
        # Create new
        c1 = db.get_or_create_collection("test", dimension=128)
        assert c1.name == "test"
        
        # Get existing
        c2 = db.get_or_create_collection("test", dimension=128)
        assert c1 is c2
    
    def test_delete_collection(self, db):
        """Test deleting a collection."""
        db.create_collection("test", dimension=128)
        
        result = db.delete_collection("test")
        
        assert result is True
        assert "test" not in db
    
    def test_delete_not_found(self, db):
        """Test deleting non-existent collection."""
        result = db.delete_collection("nonexistent")
        assert result is False
    
    def test_list_collections(self, db):
        """Test listing collections."""
        db.create_collection("a", dimension=10)
        db.create_collection("b", dimension=20)
        db.create_collection("c", dimension=30)
        
        names = db.list_collections()
        
        assert set(names) == {"a", "b", "c"}
    
    def test_indexing(self, db):
        """Test collection access via indexing."""
        db.create_collection("test", dimension=128)
        
        collection = db["test"]
        
        assert collection.name == "test"


class TestDatabaseShortcuts:
    """Shortcut method tests."""
    
    @pytest.fixture
    def db(self):
        db = VectorDB()
        db.create_collection("test", dimension=10)
        return db
    
    def test_add_shortcut(self, db):
        """Test add shortcut."""
        db.add("test", "vec1", np.random.randn(10), {"key": "value"})
        
        assert len(db["test"]) == 1
    
    def test_search_shortcut(self, db):
        """Test search shortcut."""
        for i in range(10):
            db.add("test", f"vec{i}", np.random.randn(10))
        
        results = db.search("test", np.random.randn(10), k=5)
        
        assert len(results) == 5
    
    def test_delete_shortcut(self, db):
        """Test delete shortcut."""
        db.add("test", "vec1", np.random.randn(10))
        
        result = db.delete("test", "vec1")
        
        assert result is True
        assert len(db["test"]) == 0


class TestPersistence:
    """Persistence tests."""
    
    def test_save_and_load(self):
        """Test saving and loading database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate
            db1 = VectorDB(data_dir=tmpdir)
            collection = db1.create_collection("test", dimension=10)
            
            for i in range(50):
                collection.add(
                    f"vec{i}",
                    np.random.randn(10).astype(np.float32),
                    {"index": i},
                )
            
            db1.save()
            
            # Load
            db2 = VectorDB.load(tmpdir)
            
            assert "test" in db2
            assert len(db2["test"]) == 50
            
            # Verify data
            orig = db1["test"].get("vec25")
            loaded = db2["test"].get("vec25")
            
            assert np.allclose(orig["vector"], loaded["vector"])
            assert orig["metadata"] == loaded["metadata"]
    
    def test_save_multiple_collections(self):
        """Test saving multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = VectorDB(data_dir=tmpdir)
            
            for name in ["a", "b", "c"]:
                c = db1.create_collection(name, dimension=10)
                for i in range(10):
                    c.add(f"vec{i}", np.random.randn(10))
            
            db1.save()
            
            db2 = VectorDB.load(tmpdir)
            
            assert set(db2.list_collections()) == {"a", "b", "c"}
            for name in ["a", "b", "c"]:
                assert len(db2[name]) == 10


class TestDatabaseInfo:
    """Database info tests."""
    
    def test_info(self):
        """Test database info."""
        db = VectorDB()
        
        for name, dim in [("a", 10), ("b", 20)]:
            c = db.create_collection(name, dimension=dim)
            for i in range(10):
                c.add(f"vec{i}", np.random.randn(dim))
        
        info = db.info()
        
        assert info["collection_count"] == 2
        assert info["total_vectors"] == 20
        assert "a" in info["collections"]
        assert "b" in info["collections"]