"""
Configuration module for VectorDB.

This module provides configuration management including
loading settings from YAML files and environment variables.

Example:
    >>> from config import Settings, load_config
    >>> 
    >>> # Load default config
    >>> settings = load_config()
    >>> 
    >>> # Access settings
    >>> print(settings.dimension)
    >>> print(settings.index_type)
"""

from .settings import (
    Settings,
    IVFConfig,
    HNSWConfig,
    StorageConfig,
    CacheConfig,
    load_config,
    get_default_config_path,
)

__all__ = [
    "Settings",
    "IVFConfig",
    "HNSWConfig",
    "StorageConfig",
    "CacheConfig",
    "load_config",
    "get_default_config_path",
]
