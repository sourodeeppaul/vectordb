# VectorDB API Reference

Complete API documentation for VectorDB.

## Core Classes

### VectorDB

Main database class for managing collections.

```python
from vectordb import VectorDB

db = VectorDB(data_dir="./data")
```

#### Methods

| Method | Description |
|--------|-------------|
| `create_collection(name, dimension, **kwargs)` | Create a new collection |
| `get_collection(name)` | Get existing collection |
| `delete_collection(name)` | Delete a collection |
| `list_collections()` | List all collection names |
| `save(path=None)` | Persist database to disk |
| `load(path)` | Load database from disk |
| `info()` | Get database statistics |

#### Example

```python
# Create database
db = VectorDB(data_dir="./vectordb_data")

# Create collection
collection = db.create_collection(
    name="documents",
    dimension=384,
    metric="cosine",
    index_type="hnsw"
)

# List collections
names = db.list_collections()  # ["documents"]

# Get collection
col = db.get_collection("documents")

# Save to disk
db.save()
```

---

### Collection

Vector collection for storing and searching vectors.

#### Methods

| Method | Description |
|--------|-------------|
| `add(id, vector, metadata=None)` | Add single vector |
| `add_batch(records)` | Add multiple vectors |
| `get(id)` | Get vector by ID |
| `update(id, vector=None, metadata=None)` | Update vector |
| `delete(id)` | Delete vector |
| `search(query, k=10, **kwargs)` | K-nearest neighbor search |
| `search_batch(queries, k=10)` | Batch search |
| `stats()` | Get collection statistics |

#### Example

```python
import numpy as np

# Add vectors
collection.add(
    id="doc_001",
    vector=np.random.randn(384),
    metadata={"title": "Hello World", "category": "greeting"}
)

# Batch add
from vectordb import VectorRecord
records = [
    VectorRecord(f"doc_{i}", np.random.randn(384), {"idx": i})
    for i in range(100)
]
collection.add_batch(records)

# Search
results = collection.search(
    query=np.random.randn(384),
    k=10,
    filter={"category": "greeting"}
)

for r in results:
    print(f"{r.id}: distance={r.distance:.4f}")
```

---

### VectorRecord

Container for vector data.

```python
from vectordb import VectorRecord

record = VectorRecord(
    id="unique_id",
    vector=np.array([0.1, 0.2, 0.3]),
    metadata={"key": "value"}
)
```

---

### SearchResult

Search result container.

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Vector ID |
| `distance` | float | Distance to query |
| `score` | float | Similarity score (0-1) |
| `vector` | np.ndarray | Vector data (optional) |
| `metadata` | dict | Metadata (optional) |

---

## Distance Metrics

### Available Metrics

```python
from vectordb import get_metric, euclidean, cosine_distance

# Direct use
dist = euclidean(vec_a, vec_b)

# Via registry
metric_fn = get_metric("cosine")
dist = metric_fn(vec_a, vec_b)
```

| Metric | Function | Description |
|--------|----------|-------------|
| `euclidean` | `euclidean(a, b)` | L2 distance |
| `cosine` | `cosine_distance(a, b)` | 1 - cosine similarity |
| `dot` | `dot_product(a, b)` | Negative dot product |
| `manhattan` | `manhattan(a, b)` | L1 distance |

---

## Index Types

### Creating Collections with Indexes

```python
# Flat index (brute force)
col = db.create_collection("flat", dimension=128, index_type="flat")

# IVF index
col = db.create_collection(
    "ivf", 
    dimension=128, 
    index_type="ivf",
    n_clusters=100,
    n_probe=10
)

# HNSW index
col = db.create_collection(
    "hnsw",
    dimension=128,
    index_type="hnsw",
    M=16,
    ef_construction=200
)
```

### Index Parameters

#### IVF Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | 100 | Number of clusters |
| `n_probe` | 10 | Clusters to search |
| `n_training_samples` | 10000 | Training samples |

#### HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Max connections per node |
| `ef_construction` | 200 | Construction quality |
| `ef_search` | 50 | Search quality |

---

## Storage

### Storage Backends

```python
from vectordb.storage import MemoryStorage, DiskStorage, create_storage

# Memory storage
mem = MemoryStorage(dimension=128)

# Disk storage
disk = DiskStorage("./data", dimension=128)

# Factory function
storage = create_storage("disk", path="./data", dimension=128)
```

---

## Configuration

### Loading Config

```python
from config import load_config, Settings

# Load from default path
settings = load_config()

# Load from specific file
settings = load_config("./my_config.yaml")

# Access settings
print(settings.dimension)
print(settings.metric)
```

### Config Structure

```yaml
dimension: 128
metric: euclidean
index_type: flat
normalize_vectors: false

ivf_config:
  n_clusters: 100
  n_probe: 10

hnsw_config:
  M: 16
  ef_construction: 200
  ef_search: 50

storage_config:
  data_dir: ./vectordb_data
  use_mmap: true
```

---

## REST API

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### Collections

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/collections` | Create collection |
| GET | `/collections` | List collections |
| GET | `/collections/{name}` | Get collection info |
| DELETE | `/collections/{name}` | Delete collection |

#### Vectors

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/collections/{name}/vectors` | Add vectors |
| GET | `/collections/{name}/vectors/{id}` | Get vector |
| PUT | `/collections/{name}/vectors/{id}` | Update vector |
| DELETE | `/collections/{name}/vectors/{id}` | Delete vector |

#### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/collections/{name}/search` | Search vectors |
| POST | `/collections/{name}/search/batch` | Batch search |

### Example Requests

```bash
# Create collection
curl -X POST http://localhost:8000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 128, "metric": "cosine"}'

# Add vectors
curl -X POST http://localhost:8000/api/v1/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": "v1", "vector": [0.1, 0.2, ...], "metadata": {}}]}'

# Search
curl -X POST http://localhost:8000/api/v1/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "k": 10}'
```

---

## Exceptions

| Exception | Description |
|-----------|-------------|
| `VectorDBError` | Base exception |
| `CollectionNotFoundError` | Collection doesn't exist |
| `CollectionExistsError` | Collection already exists |
| `VectorNotFoundError` | Vector ID not found |
| `VectorExistsError` | Vector ID already exists |
| `DimensionMismatchError` | Vector dimension mismatch |
| `ValidationError` | Invalid input |

```python
from vectordb import CollectionNotFoundError

try:
    col = db.get_collection("missing")
except CollectionNotFoundError:
    print("Collection not found!")
```
