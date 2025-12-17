"""
Server configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class ServerConfig:
    """Configuration for the VectorDB server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # API settings
    api_prefix: str = "/api/v1"
    docs_enabled: bool = True
    
    # Database settings
    data_dir: Optional[str] = "./vectordb_data"
    auto_save: bool = True
    auto_save_interval: int = 60
    
    # Security
    api_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Limits
    max_vectors_per_request: int = 10000
    max_dimension: int = 4096
    max_collections: int = 100
    request_timeout: int = 60
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("VECTORDB_HOST", "0.0.0.0"),
            port=int(os.getenv("VECTORDB_PORT", "8000")),
            workers=int(os.getenv("VECTORDB_WORKERS", "1")),
            data_dir=os.getenv("VECTORDB_DATA_DIR", "./vectordb_data"),
            api_key=os.getenv("VECTORDB_API_KEY"),
            log_level=os.getenv("VECTORDB_LOG_LEVEL", "INFO"),
            max_vectors_per_request=int(os.getenv("VECTORDB_MAX_VECTORS", "10000")),
        )


# Global configuration
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """Get server configuration."""
    global _config
    if _config is None:
        _config = ServerConfig.from_env()
    return _config


def set_config(config: ServerConfig) -> None:
    """Set server configuration."""
    global _config
    _config = config