# VectorDB Performance Guide

This guide covers performance optimization strategies for VectorDB.

## Quick Wins

1. **Use batch operations** instead of single inserts
2. **Choose the right index** for your data size
3. **Pre-normalize vectors** for cosine similarity
4. **Tune search parameters** based on accuracy needs

---

## Insertion Performance

### Batch vs Single Inserts

```python
# ❌ Slow - single inserts
for i, vec in enumerate(vectors):
    collection.add(f"id_{i}", vec)

# ✅ Fast - batch insert
records = [VectorRecord(f"id_{i}", vec) for i, vec in enumerate(vectors)]
collection.add_batch(records)
```

**Speedup**: 10-100x faster with batching

### Optimal Batch Size

| Vector Dimension | Recommended Batch |
|------------------|-------------------|
| 64-128 | 5000-10000 |
| 256-512 | 2000-5000 |
| 768-1024 | 500-2000 |
| 1536+ | 100-500 |

```python
BATCH_SIZE = 5000

for i in range(0, len(all_vectors), BATCH_SIZE):
    batch = all_vectors[i:i+BATCH_SIZE]
    collection.add_batch(batch)
```

---

## Search Performance

### Index Selection Impact

Benchmarks on 1M vectors, 128 dimensions:

| Index | Build Time | Search Time (k=10) | Recall@10 |
|-------|------------|-------------------|-----------|
| Flat | 1s | 150ms | 100% |
| IVF-100 | 30s | 5ms | 95% |
| HNSW | 180s | 0.5ms | 99% |

### Runtime Parameter Tuning

#### IVF: `n_probe`

```python
# Higher n_probe = better recall, slower search
results = collection.search(query, k=10, n_probe=20)
```

| n_probe | Search Time | Recall |
|---------|-------------|--------|
| 1 | 0.5ms | 60% |
| 10 | 2ms | 90% |
| 50 | 8ms | 98% |
| 100 | 15ms | 99.5% |

#### HNSW: `ef_search`

```python
# Higher ef_search = better recall, slower search
results = collection.search(query, k=10, ef_search=100)
```

| ef_search | Search Time | Recall |
|-----------|-------------|--------|
| 10 | 0.2ms | 85% |
| 50 | 0.5ms | 97% |
| 100 | 1ms | 99% |
| 200 | 2ms | 99.9% |

---

## Memory Optimization

### Storage Backend Selection

| Backend | Use Case | Memory Usage |
|---------|----------|--------------|
| Memory | Small data, testing | High (all in RAM) |
| Disk | Persistence needed | Low (disk-based) |
| MMap | Large datasets | Adaptive |

### Memory-Mapped Storage

```python
from vectordb.storage import MMapStorage

# For datasets larger than RAM
storage = MMapStorage(
    path="./large_data",
    dimension=128,
    initial_capacity=1000000
)
```

MMap benefits:
- Only loads accessed pages into RAM
- OS manages caching automatically
- Survives process restarts

### Vector Compression

```python
# Use Product Quantization for 64x compression
collection = db.create_collection(
    "compressed",
    dimension=128,
    index_type="pq",
    n_subquantizers=8,
    n_bits=8
)
```

Memory comparison (1M vectors, 128-dim):

| Format | Memory |
|--------|--------|
| float32 | 512 MB |
| PQ (8×8) | 8 MB |

---

## CPU Optimization

### SIMD Acceleration

VectorDB uses NumPy's SIMD operations. For additional speedup:

```python
# Install Numba for JIT compilation
pip install numba

# Automatically used when available
from vectordb.distance.simd import get_optimized_distance_fn

metric = get_optimized_distance_fn("euclidean")
```

### Multi-threading

```python
# Configure thread count
from config import Settings

settings = Settings(num_threads=8)
```

---

## GPU Acceleration

### Setup

```bash
# Install CuPy for CUDA support
pip install cupy-cuda11x  # Adjust for your CUDA version
```

### Usage

```python
from vectordb.distance.gpu import GPUDistanceCalculator

calc = GPUDistanceCalculator(
    metric="euclidean",
    min_vectors_for_gpu=10000  # Use GPU only for large batches
)

distances = calc.compute(query, vectors)
```

### When GPU Helps

| Dataset Size | Recommended |
|--------------|-------------|
| < 10,000 | CPU (transfer overhead) |
| 10,000 - 100,000 | Either (benchmark) |
| > 100,000 | GPU |

---

## Query Optimization

### Filter Efficiency

Filters are applied **after** the initial search. This means:

```python
# Search finds top-k candidates FIRST, then filters
results = collection.search(
    query,
    k=10,
    filter={"category": "tech"}  # Applied post-search
)
```

For better filter performance:
1. Increase `k` to compensate for filtered results
2. Use pre-filtering with separate indexes when possible

### Batch Search

```python
# ❌ Slow - sequential searches
results = [collection.search(q, k=10) for q in queries]

# ✅ Fast - batch search
results = collection.search_batch(queries, k=10)
```

---

## Benchmarking

### Built-in Benchmarks

```bash
# Run benchmarks
cd tests/benchmark
python bench_search.py
python bench_indexing.py
python bench_recall.py
```

### Custom Benchmarking

```python
import time
import numpy as np

def benchmark_search(collection, n_queries=100, k=10):
    queries = np.random.randn(n_queries, collection.dimension)
    
    start = time.time()
    for q in queries:
        collection.search(q, k=k)
    total_time = time.time() - start
    
    qps = n_queries / total_time
    latency_ms = (total_time / n_queries) * 1000
    
    print(f"QPS: {qps:.0f}")
    print(f"Latency: {latency_ms:.2f}ms")
    
    return qps, latency_ms
```

---

## Production Checklist

- [ ] Choose appropriate index type for data size
- [ ] Tune index parameters (M, ef, n_probe)
- [ ] Use batch operations for bulk inserts
- [ ] Enable memory-mapped storage for large datasets
- [ ] Configure appropriate thread count
- [ ] Set up monitoring for latency/throughput
- [ ] Plan for index rebuilds during maintenance windows

---

## Troubleshooting

### Slow Searches

1. Check index type - use HNSW for speed
2. Reduce `ef_search`/`n_probe` (trade-off with recall)
3. Ensure vectors are normalized for cosine
4. Check if filters are too restrictive

### High Memory Usage

1. Switch to disk or mmap storage
2. Use PQ compression
3. Reduce HNSW `M` parameter
4. Delete unused collections

### Poor Recall

1. Increase `ef_search`/`n_probe`
2. Increase HNSW `M` and `ef_construction`
3. Retrain IVF with more clusters
4. Consider using flat index for critical searches
