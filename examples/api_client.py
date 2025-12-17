"""
Example client for VectorDB REST API.
"""

import requests
import numpy as np
from typing import List, Dict, Any, Optional


class VectorDBClient:
    """
    Python client for VectorDB REST API.
    
    Example:
        >>> client = VectorDBClient("http://localhost:8000")
        >>> client.create_collection("docs", dimension=384, metric="cosine")
        >>> client.add_vectors("docs", [{"id": "1", "vector": [...], "metadata": {...}}])
        >>> results = client.search("docs", query_vector, k=10)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/api/v1"
        self.headers = {}
        
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def _url(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}{path}"
    
    def _request(
        self,
        method: str,
        path: str,
        json: Any = None,
        params: Dict = None,
    ) -> Dict:
        response = requests.request(
            method,
            self._url(path),
            json=json,
            params=params,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # Health & Info
    # =========================================================================
    
    def health(self) -> Dict:
        """Check server health."""
        return self._request("GET", "/health")
    
    def info(self) -> Dict:
        """Get database info."""
        return self._request("GET", "/info")
    
    # =========================================================================
    # Collections
    # =========================================================================
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "euclidean",
        index_type: str = "flat",
        **kwargs,
    ) -> Dict:
        """Create a new collection."""
        return self._request("POST", "/collections", json={
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "index_type": index_type,
            **kwargs,
        })
    
    def list_collections(self) -> List[Dict]:
        """List all collections."""
        return self._request("GET", "/collections")["collections"]
    
    def get_collection(self, name: str) -> Dict:
        """Get collection info."""
        return self._request("GET", f"/collections/{name}")
    
    def delete_collection(self, name: str) -> Dict:
        """Delete a collection."""
        return self._request("DELETE", f"/collections/{name}")
    
    # =========================================================================
    # Vectors
    # =========================================================================
    
    def add_vectors(
        self,
        collection: str,
        vectors: List[Dict],
    ) -> Dict:
        """Add vectors to a collection."""
        return self._request(
            "POST",
            f"/collections/{collection}/vectors",
            json={"vectors": vectors},
        )
    
    def add_vector(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict = None,
    ) -> Dict:
        """Add a single vector."""
        return self._request(
            "POST",
            f"/collections/{collection}/vectors/single",
            json={
                "id": id,
                "vector": vector,
                "metadata": metadata or {},
            },
        )
    
    def get_vector(
        self,
        collection: str,
        id: str,
        include_vector: bool = True,
    ) -> Dict:
        """Get a vector by ID."""
        return self._request(
            "GET",
            f"/collections/{collection}/vectors/{id}",
            params={"include_vector": include_vector},
        )
    
    def update_vector(
        self,
        collection: str,
        id: str,
        vector: List[float] = None,
        metadata: Dict = None,
    ) -> Dict:
        """Update a vector."""
        return self._request(
            "PUT",
            f"/collections/{collection}/vectors/{id}",
            json={
                "vector": vector,
                "metadata": metadata,
            },
        )
    
    def delete_vector(self, collection: str, id: str) -> Dict:
        """Delete a vector."""
        return self._request(
            "DELETE",
            f"/collections/{collection}/vectors/{id}",
        )
    
    def delete_vectors(self, collection: str, ids: List[str]) -> Dict:
        """Delete multiple vectors."""
        return self._request(
            "POST",
            f"/collections/{collection}/vectors/delete",
            json={"ids": ids},
        )
    
    # =========================================================================
    # Search
    # =========================================================================
    
    def search(
        self,
        collection: str,
        vector: List[float],
        k: int = 10,
        filter: Dict = None,
        include_vector: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict]:
        """Search for similar vectors."""
        result = self._request(
            "POST",
            f"/collections/{collection}/search",
            json={
                "vector": vector,
                "k": k,
                "filter": {"match": filter} if filter else None,
                "include_vector": include_vector,
                "include_metadata": include_metadata,
            },
        )
        return result["results"]
    
    def batch_search(
        self,
        collection: str,
        vectors: List[List[float]],
        k: int = 10,
        filter: Dict = None,
    ) -> List[List[Dict]]:
        """Batch search with multiple queries."""
        result = self._request(
            "POST",
            f"/collections/{collection}/search/batch",
            json={
                "vectors": vectors,
                "k": k,
                "filter": {"match": filter} if filter else None,
            },
        )
        return result["results"]
    
    # =========================================================================
    # Admin
    # =========================================================================
    
    def save(self, path: str = None) -> Dict:
        """Save database to disk."""
        return self._request(
            "POST",
            "/save",
            json={"path": path} if path else None,
        )


def main():
    """Example usage of VectorDB client."""
    print("=" * 60)
    print("VectorDB API Client Example")
    print("=" * 60)
    
    # Connect to server
    client = VectorDBClient("http://localhost:8000")
    
    try:
        # Check health
        print("\n1. Health check...")
        health = client.health()
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
    except requests.exceptions.ConnectionError:
        print("\n   ERROR: Server not running!")
        print("   Start server with: python -m vectordb.server")
        return
    
    # Create collection
    print("\n2. Creating collection...")
    try:
        collection = client.create_collection(
            name="demo",
            dimension=128,
            metric="cosine",
        )
        print(f"   Created: {collection['name']}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print("   Collection already exists")
        else:
            raise
    
    # Add vectors
    print("\n3. Adding vectors...")
    vectors = []
    for i in range(100):
        vectors.append({
            "id": f"doc_{i}",
            "vector": np.random.randn(128).tolist(),
            "metadata": {"index": i, "category": f"cat_{i % 5}"}
        })
    
    result = client.add_vectors("demo", vectors)
    print(f"   Added: {result['added_count']} vectors")
    
    # Search
    print("\n4. Searching...")
    query = np.random.randn(128).tolist()
    results = client.search("demo", query, k=5)
    
    print("   Top 5 results:")
    for r in results:
        print(f"   - {r['id']}: distance={r['distance']:.4f}")
    
    # Search with filter
    print("\n5. Filtered search (category=cat_0)...")
    results = client.search(
        "demo",
        query,
        k=5,
        filter={"category": "cat_0"}
    )
    
    print("   Results:")
    for r in results:
        print(f"   - {r['id']}: category={r['metadata']['category']}")
    
    # Get info
    print("\n6. Database info...")
    info = client.info()
    print(f"   Collections: {info['collection_count']}")
    print(f"   Total vectors: {info['total_vectors']}")
    
    # Clean up
    print("\n7. Cleanup...")
    client.delete_collection("demo")
    print("   Deleted demo collection")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()