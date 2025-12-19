# VectorDB 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance vector database for similarity search, built from scratch in Python. Perfect for semantic search, recommendation systems, RAG applications, and more.

## ✨ Features

- 🔍 **Multiple Index Types**: Flat (brute-force), IVF, HNSW, Product Quantization
- 📏 **Distance Metrics**: Euclidean, Cosine, Dot Product, Manhattan
- 🏷️ **Metadata Filtering**: Filter search results by custom metadata
- 💾 **Flexible Storage**: In-memory, disk-based, or memory-mapped files
- 🔄 **Full CRUD**: Create, read, update, delete operations
- ⚡ **Optimized**: NumPy vectorization, optional Numba JIT & GPU acceleration
- 🌐 **REST API**: Optional FastAPI server for HTTP access
- 📊 **Batch Operations**: Efficient bulk insert and search

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sourodeeppaul/vectordb.git
cd vectordb

# Install in development mode
pip install -e .

# With optional dependencies
pip install -e ".[dev]"      # Development tools
pip install -e ".[server]"   # REST API server
pip install -e ".[all]"      # Everything
```

## 🚀 Quick Start

```python
from vectordb import VectorDB
import numpy as np

# Create database
db = VectorDB(data_dir="./my_vectors")

# Create a collection
collection = db.create_collection(
    name="documents",
    dimension=384,
    metric="cosine"
)

# Add vectors with metadata
for i in range(1000):
    collection.add(
        id=f"doc_{i}",
        vector=np.random.randn(384).astype(np.float32),
        metadata={"category": f"cat_{i % 5}", "score": i * 0.1}
    )

# Search for similar vectors
query = np.random.randn(384).astype(np.float32)
results = collection.search(query, k=10)

for result in results:
    print(f"ID: {result.id}, Distance: {result.distance:.4f}")

# Search with metadata filter
results = collection.search(
    query, 
    k=10, 
    filter={"category": "cat_2"}
)

# Save to disk
db.save()
```

## 📚 Index Types

| Index | Best For | Accuracy | Speed | Memory |
|-------|----------|----------|-------|--------|
| **Flat** | Small datasets (<10k) | 100% | O(n) | Low |
| **IVF** | Medium datasets | ~95% | Fast | Medium |
| **HNSW** | Large datasets | ~99% | Fastest | High |
| **PQ** | Memory-constrained | ~90% | Fast | Very Low |

```python
# Create with specific index type
collection = db.create_collection(
    name="hnsw_collection",
    dimension=128,
    index_type="hnsw",
    M=16,
    ef_construction=200
)
```

## 🌐 REST API

Start the server:

```bash
# Using uvicorn
uvicorn vectordb.server.app:app --host 0.0.0.0 --port 8000

# Or using the module
python -m vectordb.server
```

Example API calls:

```bash
# Create collection
curl -X POST http://localhost:8000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 128}'

# Add vectors
curl -X POST http://localhost:8000/api/v1/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": "v1", "vector": [...]}]}'

# Search
curl -X POST http://localhost:8000/api/v1/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [...], "k": 10}'
```

## 📁 Project Structure

```
vectordb/
├── config/                 # Configuration management
├── docs/                   # Documentation
├── examples/               # Usage examples
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test suite
└── vectordb/               # Main package
    ├── core/               # Database, Collection, Vector
    ├── distance/           # Distance metrics & optimizations
    ├── index/              # Index implementations
    ├── query/              # Query processing
    ├── storage/            # Persistence layer
    ├── utils/              # Utilities
    └── server/             # REST API server
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=vectordb --cov-report=html

# Run benchmarks
python tests/benchmark/bench_search.py
```

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Indexing Guide](docs/indexing_guide.md)
- [Performance Tuning](docs/performance.md)

## 🔧 Configuration

Create a `config.yaml` file:

```yaml
dimension: 128
metric: euclidean
index_type: hnsw

hnsw_config:
  M: 16
  ef_construction: 200
  ef_search: 50

storage_config:
  data_dir: ./vectordb_data
  use_mmap: true
```

Load configuration:

```python
from config import load_config

settings = load_config("config.yaml")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [FAISS](https://github.com/facebookresearch/faiss), [Milvus](https://milvus.io/), and [Qdrant](https://qdrant.tech/)
- Built with [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [FastAPI](https://fastapi.tiangolo.com/)
