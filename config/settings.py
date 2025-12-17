"""
Configuration management for VectorDB.

Provides dataclasses for configuration and utilities
for loading settings from YAML files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import yaml


@dataclass
class IVFConfig:
    """IVF index configuration."""
    n_clusters: int = 100
    n_probe: int = 10
    n_training_samples: int = 10000
    max_iterations: int = 20


@dataclass
class HNSWConfig:
    """HNSW index configuration."""
    M: int = 16
    M_max: int = 16
    M_max0: int = 32
    ef_construction: int = 200
    ef_search: int = 50


@dataclass
class StorageConfig:
    """Storage backend configuration."""
    data_dir: str = "./vectordb_data"
    use_mmap: bool = True
    sync_writes: bool = False
    compression: Optional[Literal["lz4", "zstd"]] = None


@dataclass
class CacheConfig:
    """Cache configuration."""
    enable_cache: bool = True
    max_cache_size_mb: int = 256
    cache_policy: Literal["lru", "lfu"] = "lru"


@dataclass
class Settings:
    """
    Main settings container for VectorDB.
    
    Attributes:
        dimension: Default vector dimension
        metric: Distance metric (euclidean, cosine, dot, manhattan)
        index_type: Index type (flat, ivf, hnsw)
        normalize_vectors: Whether to normalize vectors
        ivf_config: IVF index settings
        hnsw_config: HNSW index settings
        storage_config: Storage backend settings
        cache_config: Cache settings
        batch_size: Default batch operation size
        num_threads: Number of worker threads
        log_level: Logging level
    """
    dimension: int = 128
    metric: Literal["euclidean", "cosine", "dot", "manhattan"] = "euclidean"
    index_type: Literal["flat", "ivf", "hnsw"] = "flat"
    normalize_vectors: bool = False
    
    ivf_config: IVFConfig = field(default_factory=IVFConfig)
    hnsw_config: HNSWConfig = field(default_factory=HNSWConfig)
    storage_config: StorageConfig = field(default_factory=StorageConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    
    batch_size: int = 1000
    num_threads: int = 4
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        """Create Settings from dictionary."""
        # Extract nested configs
        ivf_data = data.pop("ivf_config", {})
        hnsw_data = data.pop("hnsw_config", {})
        storage_data = data.pop("storage_config", {})
        cache_data = data.pop("cache_config", {})
        
        return cls(
            ivf_config=IVFConfig(**ivf_data),
            hnsw_config=HNSWConfig(**hnsw_data),
            storage_config=StorageConfig(**storage_data),
            cache_config=CacheConfig(**cache_data),
            **data
        )
    
    def to_dict(self) -> dict:
        """Convert Settings to dictionary."""
        from dataclasses import asdict
        return asdict(self)


def get_default_config_path() -> Path:
    """Get path to default configuration file."""
    # Check for config in current directory
    local_config = Path("./config/default_config.yaml")
    if local_config.exists():
        return local_config
    
    # Check for config relative to this file
    module_config = Path(__file__).parent / "default_config.yaml"
    if module_config.exists():
        return module_config
    
    # Check environment variable
    env_config = os.environ.get("VECTORDB_CONFIG")
    if env_config:
        return Path(env_config)
    
    return module_config


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Settings object with loaded configuration
        
    Example:
        >>> settings = load_config()
        >>> settings = load_config("./my_config.yaml")
    """
    if config_path is None:
        path = get_default_config_path()
    else:
        path = Path(config_path)
    
    if not path.exists():
        # Return default settings if no config file
        return Settings()
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        return Settings()
    
    return Settings.from_dict(data)
