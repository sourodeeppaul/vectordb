"""
Main FastAPI application for VectorDB.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from .config import get_config, ServerConfig
from .routes import create_api_router
from .middleware import RequestLoggingMiddleware
from .dependencies import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vectordb.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting VectorDB server...")
    config = get_config()
    logger.info(f"Data directory: {config.data_dir}")
    
    # Initialize database
    db = DatabaseManager.get_database()
    logger.info(f"Database initialized with {len(db.list_collections())} collections")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VectorDB server...")
    DatabaseManager.shutdown()
    logger.info("Server shutdown complete")


def create_app(config: ServerConfig = None) -> FastAPI:
    """
    Create the FastAPI application.
    
    Args:
        config: Optional server configuration
        
    Returns:
        FastAPI application instance
    """
    if config:
        from .config import set_config
        set_config(config)
    
    config = get_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="VectorDB API",
        description="""
# VectorDB - Vector Database for Similarity Search

A high-performance vector database with REST API.

## Features

- **Multiple Index Types**: Flat, HNSW, IVF
- **Distance Metrics**: Euclidean, Cosine, Dot Product
- **Metadata Filtering**: Filter search results by metadata
- **Batch Operations**: Efficient batch insert and search
- **Persistence**: Save and load databases

## Quick Start

1. Create a collection
2. Add vectors with metadata
3. Search for similar vectors

        """,
        version="0.1.0",
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url="/redoc" if config.docs_enabled else None,
        openapi_url="/openapi.json" if config.docs_enabled else None,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc) if config.log_level == "DEBUG" else None,
            }
        )
    
    # Include API routes
    api_router = create_api_router()
    app.include_router(api_router, prefix=config.api_prefix)
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "VectorDB",
            "version": "0.1.0",
            "docs": "/docs",
            "api": config.api_prefix,
        }
    
    return app


# Default app instance
app = create_app()


# Entry point for running directly
if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "vectordb.server.app:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=config.workers,
        log_level=config.log_level.lower(),
    )