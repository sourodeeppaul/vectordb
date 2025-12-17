"""
Basic usage example for VectorDB.
"""

import numpy as np
from vectordb import VectorDB, Collection


def main():
    print("=" * 60)
    print("VectorDB Basic Usage Example")
    print("=" * 60)
    
    # 1. Create database
    print("\n1. Creating database...")
    db = VectorDB()
    
    # 2. Create collection
    print("2. Creating collection...")
    collection = db.create_collection(
        name="documents",
        dimension=128,
        metric="cosine",
    )
    print(f"   Created: {collection}")
    
    # 3. Add vectors
    print("\n3. Adding vectors...")
    
    # Single add
    collection.add(
        id="doc_001",
        vector=np.random.randn(128),
        metadata={
            "title": "Introduction to VectorDB",
            "category": "tutorial",
            "year": 2024,
        }
    )
    
    # Batch add
    vectors = [
        {
            "id": f"doc_{i:03d}",
            "vector": np.random.randn(128),
            "metadata": {
                "title": f"Document {i}",
                "category": ["tutorial", "guide", "reference"][i % 3],
                "year": 2020 + (i % 5),
            }
        }
        for i in range(2, 102)
    ]
    
    result = collection.add_batch(vectors)
    print(f"   Added {result['success_count']} vectors")
    print(f"   Total in collection: {len(collection)}")
    
    # 4. Search
    print("\n4. Searching...")
    query = np.random.randn(128).astype(np.float32)
    
    results = collection.search(query, k=5)
    
    print("   Top 5 results:")
    for i, r in enumerate(results, 1):
        print(f"   {i}. {r.id}: distance={r.distance:.4f}, "
              f"category={r.metadata['category']}")
    
    # 5. Filtered search
    print("\n5. Filtered search (category='tutorial')...")
    
    results = collection.search(
        query, 
        k=5, 
        filter={"category": "tutorial"}
    )
    
    print("   Results:")
    for r in results:
        print(f"   - {r.id}: {r.metadata['title']}")
    
    # 6. Get and update
    print("\n6. Get and update...")
    
    doc = collection.get("doc_001")
    print(f"   Original: {doc['metadata']['title']}")
    
    collection.update(
        "doc_001",
        metadata={"title": "Updated: Introduction to VectorDB", "updated": True}
    )
    
    doc = collection.get("doc_001")
    print(f"   Updated: {doc['metadata']['title']}")
    
    # 7. Query by metadata
    print("\n7. Query by metadata (year >= 2023)...")
    
    docs = collection.query_by_metadata(
        filter={"year": {"$gte": 2023}},
        limit=5,
    )
    
    for doc in docs:
        print(f"   - {doc['id']}: year={doc['metadata']['year']}")
    
    # 8. Collection stats
    print("\n8. Collection statistics...")
    stats = collection.stats()
    print(f"   Vectors: {stats.vector_count}")
    print(f"   Dimension: {stats.dimension}")
    print(f"   Metric: {stats.metric}")
    print(f"   Memory: {stats.memory_usage_bytes / 1024:.1f} KB")
    
    # 9. Delete
    print("\n9. Delete operations...")
    
    deleted = collection.delete("doc_001")
    print(f"   Deleted doc_001: {deleted}")
    
    result = collection.delete_many(["doc_002", "doc_003", "nonexistent"])
    print(f"   Batch delete: {result}")
    
    print(f"   Remaining: {len(collection)} vectors")
    
    # 10. Database info
    print("\n10. Database info...")
    info = db.info()
    print(f"   Collections: {info['collection_count']}")
    print(f"   Total vectors: {info['total_vectors']}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()