"""
FastAPI dependencies for the VectorDB server.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Header, Request
from typing import Optional, Annotated
import time

from .config import get_config, ServerConfig
from ..core.database import VectorDB
from ..core.collection import Collection
from ..core.exceptions import (
    CollectionNotFoundError,
    CollectionExistsError,
    VectorNotFoundError,
)


# =============================================================================
# DATABASE SINGLETON
# =============================================================================

class DatabaseManager:
    """
    Manages the VectorDB instance.
    
    Provides a singleton database instance for the application.
    """
    
    _instance: Optional[VectorDB] = None
    _start_time: float = 0
    
    @classmethod
    def get_database(cls) -> VectorDB:
        """Get or create the database instance."""
        if cls._instance is None:
            config = get_config()
            cls._instance = VectorDB(
                data_dir=config.data_dir,
                auto_save=config.auto_save,
                auto_save_interval=config.auto_save_interval,
            )
            cls._start_time = time.time()
        return cls._instance
    
    @classmethod
    def get_uptime(cls) -> float:
        """Get server uptime in seconds."""
        if cls._start_time == 0:
            return 0
        return time.time() - cls._start_time
    
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the database."""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


def get_database() -> VectorDB:
    """Dependency to get the database instance."""
    return DatabaseManager.get_database()


def get_uptime() -> float:
    """Dependency to get server uptime."""
    return DatabaseManager.get_uptime()


# =============================================================================
# COLLECTION DEPENDENCY
# =============================================================================

async def get_collection(
    collection_name: str,
    db: VectorDB = Depends(get_database),
) -> Collection:
    """
    Dependency to get a collection by name.
    
    Raises HTTPException if collection not found.
    """
    try:
        return db.get_collection(collection_name)
    except CollectionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found"
        )


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

async def verify_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None,
    config: ServerConfig = Depends(lambda: get_config()),
) -> bool:
    """
    Verify API key if authentication is enabled.
    """
    if config.api_key is None:
        # No authentication required
        return True
    
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if x_api_key != config.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
    
    return True


# =============================================================================
# REQUEST VALIDATION
# =============================================================================

async def validate_vector_dimension(
    request: Request,
    collection: Collection = Depends(get_collection),
) -> Collection:
    """
    Validate that request vectors match collection dimension.
    """
    # This is handled in the route handlers
    return collection


# =============================================================================
# RATE LIMITING (Simple in-memory)
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests: dict = {}  # IP -> list of timestamps
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] if t > minute_ago
            ]
        else:
            self.requests[client_ip] = []
        
        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Record request
        self.requests[client_ip].append(now)
        return True


_rate_limiter = RateLimiter()


async def check_rate_limit(request: Request) -> bool:
    """
    Check rate limit for the request.
    """
    client_ip = request.client.host if request.client else "unknown"
    
    if not _rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    return True