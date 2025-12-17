"""
Retrieval-Augmented Generation (RAG) Example for VectorDB.

This example demonstrates how to build a RAG system using VectorDB
for document retrieval combined with an LLM for answer generation.

Features demonstrated:
- Document chunking and embedding
- Context retrieval with VectorDB
- LLM integration for answer generation
- Conversation history management
- Source citation

Requirements:
    pip install sentence-transformers openai tiktoken
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import re
import time

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from vectordb import VectorDatabase


@dataclass
class Document:
    """Represents a document with content and metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content[:100].encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}_chunk_{self.chunk_index}"


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    chunk: Chunk
    score: float
    
    
@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    sources: List[RetrievalResult]
    context_used: str
    tokens_used: int = 0


class TextChunker:
    """Handles document chunking with various strategies."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "recursive"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy ('fixed', 'sentence', 'recursive')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document into chunks."""
        if self.strategy == "fixed":
            return self._fixed_chunking(document)
        elif self.strategy == "sentence":
            return self._sentence_chunking(document)
        elif self.strategy == "recursive":
            return self._recursive_chunking(document)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _fixed_chunking(self, document: Document) -> List[Chunk]:
        """Simple fixed-size chunking."""
        text = document.content
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                content=chunk_text.strip(),
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                metadata={**document.metadata, "start_char": start, "end_char": end}
            ))
            
            start = end - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def _sentence_chunking(self, document: Document) -> List[Chunk]:
        """Chunk by sentences, respecting chunk size limits."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', document.content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(Chunk(
                    content=" ".join(current_chunk),
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    metadata=document.metadata
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(Chunk(
                content=" ".join(current_chunk),
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                metadata=document.metadata
            ))
        
        return chunks
    
    def _recursive_chunking(self, document: Document) -> List[Chunk]:
        """Recursively chunk using different separators."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_text(text: str, separators: List[str]) -> List[str]:
            if not separators:
                return [text]
            
            separator = separators[0]
            remaining_separators = separators[1:]
            
            if separator == "":
                # Character-level split as last resort
                return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            
            parts = text.split(separator)
            
            result = []
            current = []
            current_length = 0
            
            for part in parts:
                part_with_sep = part + separator if separator != "" else part
                
                if current_length + len(part_with_sep) > self.chunk_size:
                    if current:
                        combined = separator.join(current)
                        if len(combined) <= self.chunk_size:
                            result.append(combined)
                        else:
                            # Recursively split with finer separators
                            result.extend(split_text(combined, remaining_separators))
                        current = []
                        current_length = 0
                    
                    if len(part_with_sep) > self.chunk_size:
                        # This part alone is too big
                        result.extend(split_text(part, remaining_separators))
                    else:
                        current.append(part)
                        current_length = len(part_with_sep)
                else:
                    current.append(part)
                    current_length += len(part_with_sep)
            
            if current:
                result.append(separator.join(current))
            
            return result
        
        text_chunks = split_text(document.content, separators)
        
        return [
            Chunk(
                content=text.strip(),
                doc_id=document.doc_id,
                chunk_index=i,
                metadata=document.metadata
            )
            for i, text in enumerate(text_chunks)
            if text.strip()
        ]


class RAGSystem:
    """
    A complete RAG system combining VectorDB retrieval with LLM generation.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        db_path: Optional[str] = None,
        collection_name: str = "rag_documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            llm_model: OpenAI model for generation
            db_path: Path for VectorDB storage
            collection_name: Name of the vector collection
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.llm_model = llm_model
        self.collection_name = collection_name
        
        # Initialize embedding model
        if HAS_EMBEDDINGS:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
            self.dimension = self.embedder.get_sentence_embedding_dimension()
        else:
            print("Warning: sentence-transformers not available, using mock embeddings")
            self.embedder = None
            self.dimension = 384
        
        # Initialize chunker
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="recursive"
        )
        
        # Initialize VectorDB
        self.db = VectorDatabase(storage_path=db_path)
        
        if self.db.has_collection(collection_name):
            self.collection = self.db.get_collection(collection_name)
        else:
            self.collection = self.db.create_collection(
                name=collection_name,
                dimension=self.dimension,
                metric="cosine",
                index_type="hnsw"
            )
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Token counter
        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.encoding_for_model(llm_model)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        if self.embedder:
            return self.embedder.encode(texts, convert_to_numpy=True)
        else:
            # Mock embeddings
            return np.random.randn(len(texts), self.dimension).astype(np.float32)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate
            return len(text) // 4
    
    def add_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> int:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of documents to add
            show_progress: Whether to show progress
            
        Returns:
            Number of chunks added
        """
        all_chunks = []
        
        # Chunk all documents
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        if show_progress:
            print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Generate embeddings
        chunk_texts = [c.content for c in all_chunks]
        embeddings = self._embed(chunk_texts)
        
        # Prepare metadata
        metadata = [
            {
                "content": c.content,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                **c.metadata
            }
            for c in all_chunks
        ]
        
        # Add to collection
        ids = [c.chunk_id for c in all_chunks]
        self.collection.add(embeddings, ids=ids, metadata=metadata)
        
        if show_progress:
            print(f"Added {len(all_chunks)} chunks to the index")
        
        return len(all_chunks)
    
    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> int:
        """
        Add a single text document.
        
        Args:
            text: Document text
            metadata: Optional metadata
            doc_id: Optional document ID
            
        Returns:
            Number of chunks added
        """
        doc = Document(content=text, metadata=metadata or {}, doc_id=doc_id)
        return self.add_documents([doc], show_progress=False)
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            filter: Optional metadata filter
            min_score: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        # Embed query
        query_embedding = self._embed([query])[0]
        
        # Search
        results = self.collection.search(
            query_embedding,
            k=k,
            filter=filter
        )
        
        # Convert to RetrievalResults
        retrieval_results = []
        for r in results:
            score = 1 - r["distance"]  # Convert distance to similarity
            
            if score >= min_score:
                chunk = Chunk(
                    content=r["metadata"].get("content", ""),
                    doc_id=r["metadata"].get("doc_id", ""),
                    chunk_index=r["metadata"].get("chunk_index", 0),
                    metadata={k: v for k, v in r["metadata"].items() 
                             if k not in ["content", "doc_id", "chunk_index"]}
                )
                retrieval_results.append(RetrievalResult(chunk=chunk, score=score))
        
        return retrieval_results
    
    def _build_context(
        self,
        retrieval_results: List[RetrievalResult],
        max_tokens: int = 2000
    ) -> str:
        """Build context string from retrieval results."""
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(retrieval_results, 1):
            chunk_text = f"[Source {i}]: {result.chunk.content}"
            chunk_tokens = self._count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build the prompt for the LLM."""
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite your sources using [Source N] notation
- Be concise but thorough"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last few turns)
        for msg in self.conversation_history[-4:]:
            messages.append(msg)
        
        # Add current query with context
        user_message = f"""Context:
{context}

Question: {query}

Please answer based on the context provided above."""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        max_context_tokens: int = 2000
    ) -> RAGResponse:
        """
        Generate an answer using the LLM.
        
        Args:
            query: User query
            retrieval_results: Retrieved chunks
            max_context_tokens: Maximum tokens for context
            
        Returns:
            RAG response with answer and sources
        """
        # Build context
        context = self._build_context(retrieval_results, max_context_tokens)
        
        if not HAS_OPENAI:
            # Mock response for demo
            answer = f"[Mock Response] Based on the {len(retrieval_results)} retrieved sources, "
            answer += f"here's information relevant to your query about '{query[:50]}...'"
            
            return RAGResponse(
                answer=answer,
                sources=retrieval_results,
                context_used=context,
                tokens_used=0
            )
        
        # Build prompt
        messages = self._build_prompt(query, context)
        
        # Call LLM
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return RAGResponse(
                answer=answer,
                sources=retrieval_results,
                context_used=context,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            return RAGResponse(
                answer=f"Error generating response: {str(e)}",
                sources=retrieval_results,
                context_used=context,
                tokens_used=0
            )
    
    def query(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.3
    ) -> RAGResponse:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            filter: Optional metadata filter
            min_score: Minimum similarity score
            
        Returns:
            RAG response
        """
        # Retrieve
        retrieval_results = self.retrieve(question, k=k, filter=filter, min_score=min_score)
        
        if not retrieval_results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                context_used="",
                tokens_used=0
            )
        
        # Generate
        return self.generate_answer(question, retrieval_results)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_chunks": self.collection.count(),
            "conversation_turns": len(self.conversation_history) // 2,
            "embedding_model": "sentence-transformers" if self.embedder else "mock",
            "llm_model": self.llm_model
        }
    
    def close(self):
        """Close the database connection."""
        self.db.close()


def create_sample_knowledge_base() -> List[Document]:
    """Create a sample knowledge base for demonstration."""
    documents = [
        Document(
            content="""
            VectorDB Architecture Overview
            
            VectorDB is a high-performance vector database designed for similarity search applications.
            The system consists of several key components:
            
            1. Storage Layer: Handles persistent storage of vectors and metadata using memory-mapped files
               for efficient access. Supports both in-memory and disk-based storage modes.
            
            2. Index Layer: Provides multiple indexing algorithms including Flat (brute-force),
               IVF (Inverted File Index), HNSW (Hierarchical Navigable Small World), and 
               Product Quantization for different performance/accuracy tradeoffs.
            
            3. Query Engine: Processes search queries with support for metadata filtering,
               batch queries, and query optimization.
            
            4. API Layer: Exposes both Python API and REST API for client applications.
            """,
            metadata={"topic": "architecture", "category": "technical"}
        ),
        Document(
            content="""
            HNSW Index Configuration Guide
            
            The HNSW (Hierarchical Navigable Small World) index provides excellent query performance
            with high recall. Key parameters to configure:
            
            M (default: 16): The number of bi-directional links per node. Higher values improve
            recall but increase memory usage and build time. Recommended range: 12-48.
            
            ef_construction (default: 100): The size of the dynamic candidate list during index
            construction. Higher values improve index quality but slow down building.
            Recommended range: 100-500.
            
            ef_search (default: 50): The size of the dynamic candidate list during search.
            Higher values improve recall but slow down queries. Can be tuned at query time.
            
            For most use cases, start with M=16, ef_construction=100, and adjust ef_search
            based on your recall/latency requirements.
            """,
            metadata={"topic": "hnsw", "category": "configuration"}
        ),
        Document(
            content="""
            Best Practices for Vector Search
            
            1. Normalize your vectors: For cosine similarity, pre-normalize vectors to unit length
               for faster computation.
            
            2. Choose the right index: Use Flat for small datasets (<10k vectors), HNSW for
               balanced performance, and IVF-PQ for very large datasets.
            
            3. Batch your operations: Both insertions and queries are more efficient in batches.
               Aim for batch sizes of 100-1000 vectors.
            
            4. Use metadata filtering wisely: Pre-filtering can significantly reduce the search
               space but may affect recall. Post-filtering is more accurate but slower.
            
            5. Monitor recall: Regularly evaluate your index's recall against ground truth,
               especially after significant data changes.
            """,
            metadata={"topic": "best-practices", "category": "guide"}
        ),
        Document(
            content="""
            Troubleshooting Common Issues
            
            Issue: Slow query performance
            Solutions:
            - Increase ef_search for HNSW or nprobe for IVF
            - Check if metadata filters are too broad
            - Consider using a more appropriate index type
            - Ensure vectors are properly normalized
            
            Issue: Low recall/accuracy
            Solutions:
            - Increase M and ef_construction for HNSW
            - Increase nprobe for IVF
            - Use Flat index to verify expected results
            - Check embedding quality
            
            Issue: High memory usage
            Solutions:
            - Use disk-based storage instead of in-memory
            - Consider Product Quantization for compression
            - Reduce M parameter for HNSW
            - Use lower-dimensional embeddings
            """,
            metadata={"topic": "troubleshooting", "category": "support"}
        ),
    ]
    
    return documents


def main():
    """Run the RAG example."""
    print("=" * 60)
    print("RAG (Retrieval-Augmented Generation) Example")
    print("=" * 60)
    
    # Check dependencies
    print("\nDependency status:")
    print(f"  - sentence-transformers: {'✓' if HAS_EMBEDDINGS else '✗ (using mock)'}")
    print(f"  - openai: {'✓' if HAS_OPENAI else '✗ (using mock)'}")
    print(f"  - tiktoken: {'✓' if HAS_TIKTOKEN else '✗ (using estimate)'}")
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag = RAGSystem(
        db_path=None,  # In-memory for demo
        chunk_size=400,
        chunk_overlap=50
    )
    
    # Add knowledge base
    print("\nBuilding knowledge base...")
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)
    
    print(f"\nSystem stats: {rag.get_stats()}")
    
    # Example queries
    print("\n" + "=" * 60)
    print("Example Queries")
    print("=" * 60)
    
    queries = [
        "What are the main components of VectorDB?",
        "How should I configure HNSW for better recall?",
        "What should I do if my queries are slow?",
        "What's the recommended batch size for insertions?",
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Question: {query}")
        print("─" * 60)
        
        response = rag.query(query, k=3)
        
        print(f"\nAnswer:\n{response.answer}")
        
        print(f"\nSources used ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            preview = source.chunk.content[:100].replace('\n', ' ')
            print(f"  [{i}] (score: {source.score:.3f}) {preview}...")
    
    # Interactive mode hint
    print("\n" + "=" * 60)
    print("For interactive use:")
    print("  rag = RAGSystem()")
    print("  rag.add_text('Your document content...')")
    print("  response = rag.query('Your question?')")
    print("  print(response.answer)")
    print("=" * 60)
    
    # Clean up
    rag.close()


if __name__ == "__main__":
    main()