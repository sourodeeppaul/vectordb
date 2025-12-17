"""
Semantic Search Example for VectorDB.

This example demonstrates how to build a semantic search system
using VectorDB with sentence embeddings.

Features demonstrated:
- Text embedding generation
- Storing documents with metadata
- Semantic similarity search
- Filtering by metadata

Requirements:
    pip install sentence-transformers
"""

import numpy as np
from typing import List, Dict, Any, Optional
import time

# Check for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Note: sentence-transformers not installed. Using mock embeddings.")
    print("Install with: pip install sentence-transformers")

from vectordb import VectorDatabase


class SemanticSearchEngine:
    """
    A semantic search engine built on top of VectorDB.
    
    Converts text documents to vector embeddings and enables
    similarity-based retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        db_path: Optional[str] = None,
        collection_name: str = "documents"
    ):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name: Name of the sentence-transformer model
            db_path: Path for persistent storage (None for in-memory)
            collection_name: Name of the document collection
        """
        self.model_name = model_name
        self.collection_name = collection_name
        
        # Initialize embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            self.model = None
            self.dimension = 384  # Mock dimension
        
        # Initialize VectorDB
        self.db = VectorDatabase(storage_path=db_path)
        
        # Create or get collection
        if self.db.has_collection(collection_name):
            self.collection = self.db.get_collection(collection_name)
            print(f"Loaded existing collection with {self.collection.count()} documents")
        else:
            self.collection = self.db.create_collection(
                name=collection_name,
                dimension=self.dimension,
                metric="cosine",
                index_type="hnsw",
                index_params={"M": 16, "ef_construction": 100}
            )
            print(f"Created new collection: {collection_name}")
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        if self.model:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # Mock embeddings for demo without sentence-transformers
            np.random.seed(hash(texts[0]) % 2**32)
            return np.random.randn(len(texts), self.dimension).astype(np.float32)
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the search index.
        
        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
            ids: Optional custom IDs
            batch_size: Batch size for embedding generation
            
        Returns:
            List of document IDs
        """
        all_ids = []
        n_docs = len(documents)
        
        print(f"Adding {n_docs} documents...")
        start_time = time.time()
        
        for i in range(0, n_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadata[i:i + batch_size] if metadata else None
            batch_ids = ids[i:i + batch_size] if ids else None
            
            # Add document text to metadata
            if batch_meta is None:
                batch_meta = [{} for _ in batch_docs]
            for j, doc in enumerate(batch_docs):
                batch_meta[j]["text"] = doc
                batch_meta[j]["text_length"] = len(doc)
            
            # Generate embeddings
            embeddings = self._embed(batch_docs)
            
            # Add to collection
            doc_ids = self.collection.add(
                embeddings,
                ids=batch_ids,
                metadata=batch_meta
            )
            all_ids.extend(doc_ids)
            
            if (i + batch_size) % 500 == 0 or i + batch_size >= n_docs:
                print(f"  Processed {min(i + batch_size, n_docs)}/{n_docs} documents")
        
        elapsed = time.time() - start_time
        print(f"Added {n_docs} documents in {elapsed:.2f}s ({n_docs/elapsed:.0f} docs/sec)")
        
        return all_ids
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter
            include_scores: Whether to include similarity scores
            
        Returns:
            List of matching documents with metadata and scores
        """
        # Generate query embedding
        query_embedding = self._embed([query])[0]
        
        # Search
        results = self.collection.search(
            query_embedding,
            k=k,
            filter=filter
        )
        
        # Format results
        formatted = []
        for r in results:
            doc = {
                "id": r["id"],
                "text": r["metadata"].get("text", ""),
                "metadata": {k: v for k, v in r["metadata"].items() if k != "text"}
            }
            if include_scores:
                # Convert distance to similarity score (for cosine)
                doc["score"] = 1 - r["distance"]
            formatted.append(doc)
        
        return formatted
    
    def search_with_context(
        self,
        query: str,
        k: int = 5,
        context_size: int = 500
    ) -> str:
        """
        Search and format results as context for LLM.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            context_size: Max characters per document excerpt
            
        Returns:
            Formatted context string
        """
        results = self.search(query, k=k)
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            text = doc["text"][:context_size]
            if len(doc["text"]) > context_size:
                text += "..."
            context_parts.append(f"[{i}] {text}")
        
        return "\n\n".join(context_parts)
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        try:
            result = self.collection.get(doc_id)
            return {
                "id": doc_id,
                "text": result["metadata"].get("text", ""),
                "metadata": result["metadata"]
            }
        except Exception:
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.collection.delete(doc_id)
            return True
        except Exception:
            return False
    
    def count(self) -> int:
        """Get total number of documents."""
        return self.collection.count()
    
    def close(self):
        """Close the database connection."""
        self.db.close()


def create_sample_documents() -> tuple:
    """Create sample documents for demonstration."""
    documents = [
        # Technology
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Python is a popular programming language known for its simplicity and versatility.",
        "Neural networks are computing systems inspired by biological neural networks in the brain.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Kubernetes is an open-source container orchestration platform for automating deployment.",
        
        # Science
        "The human genome contains approximately 3 billion base pairs of DNA.",
        "Quantum mechanics describes the behavior of matter at atomic and subatomic scales.",
        "Climate change is causing rising sea levels and more extreme weather events.",
        "CRISPR is a revolutionary gene-editing technology that allows precise DNA modifications.",
        "Black holes are regions of spacetime where gravity is so strong nothing can escape.",
        
        # Business
        "Agile methodology emphasizes iterative development and customer collaboration.",
        "Venture capital firms invest in early-stage companies with high growth potential.",
        "Supply chain management optimizes the flow of goods from suppliers to customers.",
        "Customer relationship management (CRM) systems help businesses manage interactions.",
        "Digital transformation involves integrating digital technology into all business areas.",
        
        # Health
        "Regular exercise improves cardiovascular health and reduces disease risk.",
        "Meditation and mindfulness practices can reduce stress and improve mental health.",
        "Vaccines work by training the immune system to recognize and fight pathogens.",
        "Sleep is essential for memory consolidation and physical recovery.",
        "Nutrition plays a crucial role in maintaining overall health and preventing disease.",
    ]
    
    categories = (
        ["technology"] * 5 +
        ["science"] * 5 +
        ["business"] * 5 +
        ["health"] * 5
    )
    
    metadata = [
        {"category": cat, "doc_index": i}
        for i, cat in enumerate(categories)
    ]
    
    return documents, metadata


def main():
    """Run the semantic search example."""
    print("=" * 60)
    print("Semantic Search Example")
    print("=" * 60)
    
    # Create search engine (in-memory for demo)
    engine = SemanticSearchEngine(db_path=None)
    
    # Add sample documents
    documents, metadata = create_sample_documents()
    engine.add_documents(documents, metadata=metadata)
    
    print(f"\nTotal documents indexed: {engine.count()}")
    
    # Example searches
    print("\n" + "=" * 60)
    print("Search Examples")
    print("=" * 60)
    
    # Basic search
    print("\n1. Basic search: 'machine learning and AI'")
    print("-" * 40)
    results = engine.search("machine learning and AI", k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:80]}...")
    
    # Search with filter
    print("\n2. Filtered search: 'health' in category='health'")
    print("-" * 40)
    results = engine.search(
        "staying healthy",
        k=3,
        filter={"category": "health"}
    )
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:80]}...")
    
    # Different query
    print("\n3. Search: 'startup funding and investment'")
    print("-" * 40)
    results = engine.search("startup funding and investment", k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] ({r['metadata']['category']}) {r['text'][:60]}...")
    
    # Context for RAG
    print("\n4. Get context for RAG: 'How do vaccines work?'")
    print("-" * 40)
    context = engine.search_with_context("How do vaccines work?", k=2)
    print(context)
    
    # Clean up
    engine.close()
    
    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()