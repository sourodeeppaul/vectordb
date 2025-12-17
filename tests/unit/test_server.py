"""
Tests for VectorDB REST API server.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import tempfile
import shutil

from vectordb.server.app import create_app
from vectordb.server.config import ServerConfig, set_config


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def client(temp_dir):
    """Create test client."""
    config = ServerConfig(
        data_dir=temp_dir,
        docs_enabled=True,
        api_key=None,
    )
    set_config(config)
    
    app = create_app(config)
    
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Health endpoint tests."""
    
    def test_health(self, client):
        """Test health check."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestCollectionEndpoints:
    """Collection endpoint tests."""
    
    def test_create_collection(self, client):
        """Test creating a collection."""
        response = client.post(
            "/api/v1/collections",
            json={
                "name": "test_collection",
                "dimension": 128,
                "metric": "cosine",
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test_collection"
        assert data["dimension"] == 128
    
    def test_create_duplicate_collection(self, client):
        """Test creating duplicate collection."""
        # Create first
        client.post(
            "/api/v1/collections",
            json={"name": "test", "dimension": 64}
        )
        
        # Create duplicate
        response = client.post(
            "/api/v1/collections",
            json={"name": "test", "dimension": 64}
        )
        
        assert response.status_code == 409
    
    def test_list_collections(self, client):
        """Test listing collections."""
        # Create some collections
        for i in range(3):
            client.post(
                "/api/v1/collections",
                json={"name": f"collection_{i}", "dimension": 64}
            )
        
        response = client.get("/api/v1/collections")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["collections"]) == 3
    
    def test_get_collection(self, client):
        """Test getting collection info."""
        client.post(
            "/api/v1/collections",
            json={"name": "test", "dimension": 128}
        )
        
        response = client.get("/api/v1/collections/test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test"
        assert data["dimension"] == 128
    
    def test_get_nonexistent_collection(self, client):
        """Test getting nonexistent collection."""
        response = client.get("/api/v1/collections/nonexistent")
        
        assert response.status_code == 404
    
    def test_delete_collection(self, client):
        """Test deleting collection."""
        client.post(
            "/api/v1/collections",
            json={"name": "to_delete", "dimension": 64}
        )
        
        response = client.delete("/api/v1/collections/to_delete")
        
        assert response.status_code == 200
        
        # Verify deleted
        response = client.get("/api/v1/collections/to_delete")
        assert response.status_code == 404


class TestVectorEndpoints:
    """Vector endpoint tests."""
    
    @pytest.fixture
    def collection(self, client):
        """Create a test collection."""
        client.post(
            "/api/v1/collections",
            json={"name": "vectors", "dimension": 4}
        )
        return "vectors"
    
    def test_add_single_vector(self, client, collection):
        """Test adding a single vector."""
        response = client.post(
            f"/api/v1/collections/{collection}/vectors/single",
            json={
                "id": "vec1",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "metadata": {"key": "value"}
            }
        )
        
        assert response.status_code == 201
    
    def test_add_batch_vectors(self, client, collection):
        """Test adding multiple vectors."""
        response = client.post(
            f"/api/v1/collections/{collection}/vectors",
            json={
                "vectors": [
                    {"id": f"vec{i}", "vector": [0.1 * i] * 4}
                    for i in range(10)
                ]
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["added_count"] == 10
    
    def test_get_vector(self, client, collection):
        """Test getting a vector."""
        # Add vector
        client.post(
            f"/api/v1/collections/{collection}/vectors/single",
            json={"id": "vec1", "vector": [1, 2, 3, 4]}
        )
        
        # Get vector
        response = client.get(
            f"/api/v1/collections/{collection}/vectors/vec1"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "vec1"
        assert data["vector"] == [1, 2, 3, 4]
    
    def test_get_nonexistent_vector(self, client, collection):
        """Test getting nonexistent vector."""
        response = client.get(
            f"/api/v1/collections/{collection}/vectors/nonexistent"
        )
        
        assert response.status_code == 404
    
    def test_update_vector(self, client, collection):
        """Test updating a vector."""
        # Add vector
        client.post(
            f"/api/v1/collections/{collection}/vectors/single",
            json={"id": "vec1", "vector": [1, 2, 3, 4]}
        )
        
        # Update
        response = client.put(
            f"/api/v1/collections/{collection}/vectors/vec1",
            json={"metadata": {"updated": True}}
        )
        
        assert response.status_code == 200
    
    def test_delete_vector(self, client, collection):
        """Test deleting a vector."""
        # Add vector
        client.post(
            f"/api/v1/collections/{collection}/vectors/single",
            json={"id": "vec1", "vector": [1, 2, 3, 4]}
        )
        
        # Delete
        response = client.delete(
            f"/api/v1/collections/{collection}/vectors/vec1"
        )
        
        assert response.status_code == 200
        
        # Verify deleted
        response = client.get(
            f"/api/v1/collections/{collection}/vectors/vec1"
        )
        assert response.status_code == 404


class TestSearchEndpoints:
    """Search endpoint tests."""
    
    @pytest.fixture
    def populated_collection(self, client):
        """Create and populate a test collection."""
        # Create collection
        client.post(
            "/api/v1/collections",
            json={"name": "search_test", "dimension": 4}
        )
        
        # Add vectors
        vectors = [
            {"id": f"vec{i}", "vector": [float(i)] * 4, "metadata": {"i": i}}
            for i in range(100)
        ]
        client.post(
            "/api/v1/collections/search_test/vectors",
            json={"vectors": vectors}
        )
        
        return "search_test"
    
    def test_search(self, client, populated_collection):
        """Test basic search."""
        response = client.post(
            f"/api/v1/collections/{populated_collection}/search",
            json={
                "vector": [50.0, 50.0, 50.0, 50.0],
                "k": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 5
        assert "search_time_ms" in data
    
    def test_search_with_filter(self, client, populated_collection):
        """Test search with metadata filter."""
        response = client.post(
            f"/api/v1/collections/{populated_collection}/search",
            json={
                "vector": [50.0, 50.0, 50.0, 50.0],
                "k": 10,
                "filter": {"match": {"i": 50}}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should find the matching vector
        assert any(r["id"] == "vec50" for r in data["results"])
    
    def test_search_include_vector(self, client, populated_collection):
        """Test search with vectors included."""
        response = client.post(
            f"/api/v1/collections/{populated_collection}/search",
            json={
                "vector": [50.0, 50.0, 50.0, 50.0],
                "k": 3,
                "include_vector": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert all(r["vector"] is not None for r in data["results"])
    
    def test_batch_search(self, client, populated_collection):
        """Test batch search."""
        response = client.post(
            f"/api/v1/collections/{populated_collection}/search/batch",
            json={
                "vectors": [
                    [10.0, 10.0, 10.0, 10.0],
                    [50.0, 50.0, 50.0, 50.0],
                    [90.0, 90.0, 90.0, 90.0],
                ],
                "k": 3
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert data["total_queries"] == 3


class TestAdminEndpoints:
    """Admin endpoint tests."""
    
    def test_database_info(self, client):
        """Test database info."""
        response = client.get("/api/v1/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "collection_count" in data
    
    def test_save_database(self, client):
        """Test saving database."""
        # Create some data
        client.post(
            "/api/v1/collections",
            json={"name": "save_test", "dimension": 64}
        )
        
        # Save
        response = client.post("/api/v1/save")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "vectors_total" in data