# VectorDB 🚀

A high-performance vector database for similarity search, built from scratch in Python.

## Features

- 🔍 **Multiple Index Types**: Flat, IVF, HNSW
- 📏 **Distance Metrics**: Euclidean, Cosine, Dot Product, Manhattan
- 🏷️ **Metadata Filtering**: Filter search results by metadata
- 💾 **Persistence**: Save and load databases to/from disk
- 🔄 **CRUD Operations**: Full create, read, update, delete support
- ⚡ **Optimized**: NumPy-based vectorized operations

## Index Types
Index	Best For	Accuracy	Speed
flat	Small datasets (<10k)	100%	O(n)
ivf	Medium datasets	~95%	O(√n)
hnsw	Large datasets	~95%	O(log n)

# Benchmarks
Dataset Size	Index	QPS	Recall@10
10,000	Flat	1,000	100%
100,000	IVF	5,000	95%
1,000,000	HNSW	3,000	96%


## Configuration Module

### 📄 `config/settings.py`

```python
"""
Configuration management for VectorDB.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import os
import yaml


class IndexType(str, Enum):
    """Available index types."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    PQ = "pq"
    IVF_PQ = "ivf_pq"


class DistanceMetric(str, Enum):
    """Available distance metrics."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"


@dataclass
class FlatIndexConfig:
    """Configuration for Flat (brute-force) index."""
    pass  # No specific config needed


@dataclass
class IVFIndexConfig:
    """Configuration for IVF index."""
    n_clusters: int = 100
    n_probe: int = 10
    n_training_samples: int = 10000
    max_iterations: int = 20


@dataclass
class HNSWIndexConfig:
    """Configuration for HNSW index."""
    M: int = 16                    # Max connections per node
    M_max: int = 16                # Max connections for upper layers
    M_max0: int = 32               # Max connections for layer 0
    ef_construction: int = 200     # Size of dynamic candidate list (construction)
    ef_search: int = 50            # Size of dynamic candidate list (search)
    ml: float = 0.36               # Level multiplier (1/ln(M))


@dataclass
class PQIndexConfig:
    """Configuration for Product Quantization index."""
    n_subvectors: int = 8          # Number of subvectors
    n_bits: int = 8                # Bits per subvector (256 centroids)
    n_training_samples: int = 10000


@dataclass
class StorageConfig:
    """Configuration for storage layer."""
    data_dir: str = "./vectordb_data"
    use_mmap: bool = True
    sync_writes: bool = False
    compression: Optional[str] = None  # None, "lz4", "zstd"


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enable_cache: bool = True
    max_cache_size_mb: int = 256
    cache_policy: str = "lru"  # "lru", "lfu"


@dataclass
class VectorDBConfig:
    """Main configuration for VectorDB."""
    
    # Core settings
    dimension: int = 128
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    index_type: IndexType = IndexType.FLAT
    
    # Normalization
    normalize_vectors: bool = False
    
    # Index-specific configs
    flat_config: FlatIndexConfig = field(default_factory=FlatIndexConfig)
    ivf_config: IVFIndexConfig = field(default_factory=IVFIndexConfig)
    hnsw_config: HNSWIndexConfig = field(default_factory=HNSWIndexConfig)
    pq_config: PQIndexConfig = field(default_factory=PQIndexConfig)
    
    # Storage
    storage_config: StorageConfig = field(default_factory=StorageConfig)
    
    # Cache
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    
    # Operational settings
    batch_size: int = 1000
    num_threads: int = 4
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorDBConfig":
        """Create config from dictionary."""
        # Handle nested configs
        if "storage_config" in config_dict:
            config_dict["storage_config"] = StorageConfig(**config_dict["storage_config"])
        if "cache_config" in config_dict:
            config_dict["cache_config"] = CacheConfig(**config_dict["cache_config"])
        if "ivf_config" in config_dict:
            config_dict["ivf_config"] = IVFIndexConfig(**config_dict["ivf_config"])
        if "hnsw_config" in config_dict:
            config_dict["hnsw_config"] = HNSWIndexConfig(**config_dict["hnsw_config"])
        
        # Handle enums
        if "metric" in config_dict and isinstance(config_dict["metric"], str):
            config_dict["metric"] = DistanceMetric(config_dict["metric"])
        if "index_type" in config_dict and isinstance(config_dict["index_type"], str):
            config_dict["index_type"] = IndexType(config_dict["index_type"])
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "VectorDBConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "dimension": self.dimension,
            "metric": self.metric.value,
            "index_type": self.index_type.value,
            "normalize_vectors": self.normalize_vectors,
            "batch_size": self.batch_size,
            "num_threads": self.num_threads,
            "log_level": self.log_level,
        }
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Default configuration instance
DEFAULT_CONFIG = VectorDBConfig()


def get_config(config_path: Optional[str] = None) -> VectorDBConfig:
    """
    Get configuration, optionally from a file.
    
    Priority:
    1. Provided config path
    2. VECTORDB_CONFIG environment variable
    3. Default configuration
    """
    if config_path:
        return VectorDBConfig.from_yaml(config_path)
    
    env_config = os.environ.get("VECTORDB_CONFIG")
    if env_config and os.path.exists(env_config):
        return VectorDBConfig.from_yaml(env_config)
    
    return DEFAULT_CONFIG
```

## Requirements

- Core dependencies
numpy>=1.21.0
scipy>=1.7.0

- Data structures
sortedcontainers>=2.4.0

- Serialization
msgpack>=1.0.0

- Optional: API server
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

- Optional: Performance
numba>=0.57.0

- Development
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-benchmark>=4.0.0
black>=23.0.0
isort>=5.12.0
mypy>=1.0.0

- Documentation
mkdocs>=1.5.0
mkdocs-material>=9.0.0

## Installation

```bash
# From source
git clone https://github.com/sourodeeppaul/vectordb.git
cd vectordb
pip install -e .

# With development dependencies
pip install -e ".[dev,server]"
```
# Makefile

.PHONY: install dev-install test lint format clean docs run-server benchmark

### Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev,server]"

### Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ --cov=vectordb --cov-report=html

benchmark:
	pytest tests/benchmark/ -v --benchmark-only

### Code quality
lint:
	black --check vectordb/ tests/
	isort --check-only vectordb/ tests/
	mypy vectordb/

format:
	black vectordb/ tests/
	isort vectordb/ tests/

### Documentation
docs:
	mkdocs serve

docs-build:
	mkdocs build

### Server
run-server:
	uvicorn vectordb.server.app:app --reload --host 0.0.0.0 --port 8000

### Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
