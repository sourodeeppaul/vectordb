# VectorDB Architecture

This document describes the high-level architecture of VectorDB.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        VectorDB                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Server     │    │    Core      │    │   Storage    │       │
│  │  (FastAPI)   │───▶│  (Database)  │───▶│  (Backend)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Routes     │    │ Collections  │    │  Memory/Disk │       │
│  │   Models     │    │   Indexes    │    │    MMap      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### Database (`vectordb/core/database.py`)

The main entry point for VectorDB. Manages collections and provides a unified API.

```python
from vectordb import VectorDB

db = VectorDB(data_dir="./data")
collection = db.create_collection("my_vectors", dimension=128)
```

### Collection (`vectordb/core/collection.py`)

Manages vectors within a collection, including CRUD operations and search.

**Responsibilities:**
- Vector storage and retrieval
- Index management
- Metadata handling
- Search execution

### Vector (`vectordb/core/vector.py`)

Represents individual vectors with their metadata.

```python
@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    metadata: Optional[dict]
```

## Index Layer

### Index Types

| Index | Use Case | Memory | Speed | Accuracy |
|-------|----------|--------|-------|----------|
| Flat | Small datasets (<10k) | Low | O(n) | Exact |
| IVF | Medium datasets | Medium | Fast | ~95% |
| HNSW | Large datasets | High | Fastest | ~99% |
| PQ | Memory-constrained | Very Low | Fast | ~90% |

### Index Hierarchy

```
BaseIndex (abstract)
├── FlatIndex        # Brute force
├── IVFIndex         # Inverted file
├── HNSWIndex        # Graph-based
├── PQIndex          # Product quantization
└── HybridIndex      # IVF + PQ
```

## Storage Layer

### Storage Backends

1. **MemoryStorage**: Fast, volatile, for testing
2. **DiskStorage**: Persistent, with optional compression
3. **MMapStorage**: Memory-mapped for large datasets

### Data Flow

```
add_vector() ─▶ validate ─▶ normalize ─▶ index ─▶ storage
                  │
search() ◀────────┴───────── query ◀─── index ◀─── storage
```

## Distance Metrics

Located in `vectordb/distance/`:

- **Euclidean (L2)**: Default, geometric distance
- **Cosine**: Angular similarity (normalized)
- **Dot Product**: For inner product search
- **Manhattan (L1)**: Taxicab distance

### Optimizations

- `simd.py`: Numba JIT-compiled functions
- `gpu.py`: CuPy GPU acceleration (optional)

## Query Processing

### Query Pipeline

```
Query ─▶ Parser ─▶ Planner ─▶ Executor ─▶ Results
                      │
                      └─▶ Filter Application
```

### Components

1. **Parser**: Validates and parses query parameters
2. **Planner**: Optimizes query execution plan
3. **Executor**: Runs the query against indexes
4. **Filters**: Applies metadata filtering

## Server Layer

Optional REST API built with FastAPI.

### Endpoints

- `/collections`: Collection management
- `/vectors`: Vector CRUD operations
- `/search`: Similarity search
- `/admin`: Health, metrics, maintenance

## Configuration

### Hierarchy

1. Default config (`config/default_config.yaml`)
2. Environment variables
3. Runtime parameters

### Key Settings

```yaml
dimension: 128
metric: euclidean
index_type: flat
storage_config:
  data_dir: ./vectordb_data
  use_mmap: true
```

## Thread Safety

VectorDB uses locks for thread-safe operations:

- Collection-level locks for write operations
- Read operations are lock-free when possible
- Index operations may acquire read/write locks

## Performance Considerations

1. **Batch operations** are more efficient than single operations
2. **Index choice** significantly impacts search speed
3. **Normalization** should be done once, at insertion
4. **Metadata filters** are applied post-search (affects recall)
