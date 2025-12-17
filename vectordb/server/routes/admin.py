"""
Admin and database management endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
import time

from ..models import (
    HealthResponse,
    DatabaseInfoResponse,
    SaveRequest,
    SaveResponse,
    SuccessResponse,
    ErrorResponse,
)
from ..dependencies import get_database, get_uptime, verify_api_key
from ..config import get_config
from ...core.database import VectorDB

router = APIRouter()

# Server start time
_start_time = time.time()
_version = "0.1.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the server is healthy and running.",
)
async def health_check(
    uptime: float = Depends(get_uptime),
):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=_version,
        uptime_seconds=uptime,
    )


@router.get(
    "/info",
    response_model=DatabaseInfoResponse,
    summary="Database info",
    description="Get detailed information about the database.",
)
async def database_info(
    db: VectorDB = Depends(get_database),
    uptime: float = Depends(get_uptime),
):
    """Get database information."""
    info = db.info()
    config = get_config()
    
    return DatabaseInfoResponse(
        version=_version,
        collection_count=info["collection_count"],
        total_vectors=info["total_vectors"],
        total_memory_mb=info["total_memory_mb"],
        data_dir=config.data_dir,
        uptime_seconds=uptime,
        collections=info["collections"],
    )


@router.post(
    "/save",
    response_model=SaveResponse,
    summary="Save database",
    description="Persist the database to disk.",
)
async def save_database(
    request: SaveRequest = None,
    db: VectorDB = Depends(get_database),
    _auth: bool = Depends(verify_api_key),
):
    """Save database to disk."""
    try:
        path = request.path if request and request.path else None
        db.save(path)
        
        config = get_config()
        save_path = path or config.data_dir
        
        return SaveResponse(
            path=save_path,
            collections_saved=len(db.list_collections()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/compact",
    response_model=SuccessResponse,
    summary="Compact database",
    description="Compact storage to reclaim space from deleted vectors.",
)
async def compact_database(
    db: VectorDB = Depends(get_database),
    _auth: bool = Depends(verify_api_key),
):
    """Compact database storage."""
    db.save()
    
    return SuccessResponse(
        message="Database compacted successfully"
    )


@router.get(
    "/metrics",
    summary="Get metrics",
    description="Get server and database metrics.",
)
async def get_metrics(
    db: VectorDB = Depends(get_database),
    uptime: float = Depends(get_uptime),
):
    """Get server metrics in Prometheus format or JSON."""
    info = db.info()
    
    return {
        "uptime_seconds": uptime,
        "collections_total": info["collection_count"],
        "vectors_total": info["total_vectors"],
        "memory_bytes": int(info["total_memory_mb"] * 1024 * 1024),
        "requests_total": 0,
        "collections": {
            name: {
                "vectors": data.get("vector_count", 0),
                "memory_bytes": data.get("memory_usage_bytes", 0),
            }
            for name, data in info["collections"].items()
        }
    }
