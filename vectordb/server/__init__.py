"""
VectorDB REST API Server.

A FastAPI-based REST API for vector database operations.

Quick Start:
    >>> from vectordb.server import create_app, run_server
    >>> 
    >>> # Create and run server
    >>> app = create_app()
    >>> run_server(app, host="0.0.0.0", port=8000)

Or using command line:
    $ python -m vectordb.server --host 0.0.0.0 --port 8000
    
Or with uvicorn:
    $ uvicorn vectordb.server:app --reload
"""

from .app import create_app, app
from .config import ServerConfig, get_config
from .models import (
    # Collections
    CreateCollectionRequest,
    CollectionResponse,
    CollectionListResponse,
    # Vectors
    AddVectorRequest,
    AddVectorsRequest,
    VectorResponse,
    UpdateVectorRequest,
    # Search
    SearchRequest,
    SearchResponse,
    BatchSearchRequest,
    BatchSearchResponse,
    # Common
    SuccessResponse,
    ErrorResponse,
)

__all__ = [
    # App
    "create_app",
    "app",
    "run_server",
    # Config
    "ServerConfig",
    "get_config",
    # Models
    "CreateCollectionRequest",
    "CollectionResponse",
    "CollectionListResponse",
    "AddVectorRequest",
    "AddVectorsRequest",
    "VectorResponse",
    "UpdateVectorRequest",
    "SearchRequest",
    "SearchResponse",
    "BatchSearchRequest",
    "BatchSearchResponse",
    "SuccessResponse",
    "ErrorResponse",
]


def run_server(
    app=None,
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info",
):
    """
    Run the VectorDB server.
    
    Args:
        app: FastAPI application (creates default if None)
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        workers: Number of worker processes
        log_level: Logging level
    """
    import uvicorn
    
    if app is None:
        app = create_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
    )