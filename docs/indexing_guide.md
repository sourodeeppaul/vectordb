# VectorDB Indexing Guide

This guide explains the different index types available in VectorDB and how to choose the right one for your use case.

## Index Overview

VectorDB supports multiple index types optimized for different scenarios:

| Index | Best For | Accuracy | Speed | Memory |
|-------|----------|----------|-------|--------|
| **Flat** | Small datasets, exact search | 100% | Slow | Low |
| **IVF** | Medium datasets, balanced | ~95% | Fast | Medium |
| **HNSW** | Large datasets, speed critical | ~99% | Fastest | High |
| **PQ** | Memory constrained | ~90% | Fast | Very Low |
| **Hybrid** | Large + memory constrained | ~93% | Fast | Low |

## Flat Index

### Description

Brute-force linear scan. Compares query against every vector.

### When to Use

- Dataset < 10,000 vectors
- Need 100% accurate results
- Don't need real-time performance

### Configuration

```python
collection = db.create_collection(
    name="exact_search",
    dimension=128,
    index_type="flat"
)
```

### Complexity

- Build: O(n)
- Search: O(n × d) where d = dimension

---

## IVF (Inverted File Index)

### Description

Clusters vectors using k-means. Only searches nearby clusters.

### When to Use

- Dataset: 10,000 - 1,000,000 vectors
- Acceptable ~95% recall
- Need balance of speed and accuracy

### How It Works

1. **Training**: Clusters vectors into `n_clusters` groups
2. **Insertion**: Assigns vectors to nearest cluster
3. **Search**: Only searches `n_probe` nearest clusters

### Configuration

```python
collection = db.create_collection(
    name="ivf_index",
    dimension=128,
    index_type="ivf",
    n_clusters=100,    # Number of clusters
    n_probe=10,        # Clusters to search
)
```

### Parameter Tuning

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `n_clusters` | Faster build, slower search | Slower build, faster search |
| `n_probe` | Faster search, lower recall | Slower search, higher recall |

**Rules of thumb:**
- `n_clusters` ≈ √(n_vectors)
- `n_probe` ≈ 1-10% of `n_clusters`

### Example

```python
# For 100,000 vectors
collection = db.create_collection(
    name="my_ivf",
    dimension=384,
    index_type="ivf",
    n_clusters=316,  # √100000 ≈ 316
    n_probe=32       # ~10% of clusters
)
```

---

## HNSW (Hierarchical Navigable Small World)

### Description

Graph-based index with hierarchical layers. Fast approximate search.

### When to Use

- Dataset: 100,000+ vectors
- Need fastest possible search
- Memory is not a concern
- High accuracy required (~99%)

### How It Works

1. **Build**: Creates multi-layer graph
2. **Search**: Navigates graph from top layer down
3. Uses "small world" property for fast traversal

### Configuration

```python
collection = db.create_collection(
    name="hnsw_index",
    dimension=128,
    index_type="hnsw",
    M=16,               # Max connections per node
    ef_construction=200, # Build quality
    ef_search=50        # Search quality
)
```

### Parameter Tuning

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `M` | Graph connectivity | 12-48, higher = better recall |
| `ef_construction` | Build quality | 100-500, higher = better graph |
| `ef_search` | Search quality | 10-500, higher = better recall |

**Trade-offs:**
- Higher `M`: More memory, better recall
- Higher `ef_construction`: Slower build, better graph
- Higher `ef_search`: Slower search, better recall

### Memory Usage

```
Memory ≈ n_vectors × (M × 4 + dimension × 4) bytes
```

For 1M vectors, 128-dim, M=16:
```
1,000,000 × (16 × 4 + 128 × 4) = ~576 MB
```

---

## Product Quantization (PQ)

### Description

Compresses vectors by splitting into subvectors and quantizing.

### When to Use

- Memory is severely constrained
- Dataset is very large
- Can tolerate ~90% recall

### How It Works

1. Splits vector into `n_subquantizers` parts
2. Each part quantized to `n_bits` code
3. Distance computed using lookup tables

### Configuration

```python
collection = db.create_collection(
    name="pq_index",
    dimension=128,
    index_type="pq",
    n_subquantizers=8,  # Number of sub-vectors
    n_bits=8            # Bits per code
)
```

### Memory Savings

Original: 128 × 4 bytes = 512 bytes per vector
PQ (8 subquantizers, 8 bits): 8 bytes per vector

**64x compression!**

---

## Hybrid Index (IVF + PQ)

### Description

Combines IVF clustering with PQ compression.

### When to Use

- Very large datasets (10M+ vectors)
- Memory constrained
- Need reasonable accuracy (~93%)

### Configuration

```python
collection = db.create_collection(
    name="hybrid_index",
    dimension=128,
    index_type="hybrid",
    n_clusters=1000,
    n_subquantizers=8
)
```

---

## Choosing the Right Index

### Decision Flowchart

```
                    START
                      │
              How many vectors?
                      │
        ┌─────────────┼─────────────┐
        │             │             │
     < 10k       10k - 1M        > 1M
        │             │             │
      Flat           IVF     Memory constrained?
                      │             │
                      │     ┌───────┴───────┐
                      │    Yes             No
                      │     │               │
                   Hybrid/PQ            HNSW
```

### Quick Reference

| Scenario | Recommended Index |
|----------|-------------------|
| Prototype / testing | Flat |
| Production, balanced | IVF |
| Maximum speed | HNSW |
| Limited memory | PQ or Hybrid |
| Real-time, high accuracy | HNSW |

---

## Performance Tips

1. **Normalize vectors** for cosine similarity
2. **Batch insertions** - more efficient than single inserts
3. **Tune `ef_search`/`n_probe`** at query time based on accuracy needs
4. **Use filters sparingly** - applied post-search, affects recall
5. **Pre-allocate capacity** when dataset size is known

### Benchmarking

```python
# Available in examples/index_comparison.py
from vectordb.index import FlatIndex, HNSWIndex

# Measure build time
start = time.time()
index.build(vectors)
build_time = time.time() - start

# Measure search time
start = time.time()
results = index.search(query, k=10)
search_time = time.time() - start
```

---

## Rebuilding Indexes

Indexes can be rebuilt if:
- Parameters need tuning
- Many deletions fragmented the index
- Upgrading to different index type

```python
# Rebuild with new parameters
collection.rebuild_index(
    index_type="hnsw",
    M=32,
    ef_construction=300
)
```
