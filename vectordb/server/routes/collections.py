"""
Collection management endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..models import (
    CreateCollectionRequest,
    CollectionResponse,
    CollectionListResponse,
    SuccessResponse,
    ErrorResponse,
)
from ..dependencies import get_database, verify_api_key
from ...core.database import VectorDB
from ...core.exceptions import CollectionExistsError, CollectionNotFoundError

router = APIRouter()


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=201,
    responses={
        201: {"description": "Collection created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        409: {"model": ErrorResponse, "description": "Collection already exists"},
    },
    summary="Create a new collection",
    description="Create a new vector collection with the specified configuration.",
)
async def create_collection(
    request: CreateCollectionRequest,
    db: VectorDB = Depends(get_database),
    _auth: bool = Depends(verify_api_key),
):
    """Create a new vector collection."""
    try:
        collection = db.create_collection(
            name=request.name,
            dimension=request.dimension,
            metric=request.metric.value,
            normalize=request.normalize,
        )
        
        stats = collection.stats()
        
        return CollectionResponse(
            name=collection.name,
            dimension=collection.dimension,
            metric=collection.metric,
            index_type=request.index_type.value,
            vector_count=0,
            created_at=stats.created_at,
            updated_at=stats.updated_at,
            description=request.description,
            memory_usage_bytes=stats.memory_usage_bytes,
        )
    
    except CollectionExistsError:
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{request.name}' already exists"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "",
    response_model=CollectionListResponse,
    summary="List all collections",
    description="Get a list of all collections in the database.",
)
async def list_collections(
    db: VectorDB = Depends(get_database),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List all collections with pagination."""
    collection_names = db.list_collections()
    
    total = len(collection_names)
    paginated_names = collection_names[offset:offset + limit]
    
    collections = []
    for name in paginated_names:
        try:
            collection = db.get_collection(name)
            stats = collection.stats()
            
            collections.append(CollectionResponse(
                name=name,
                dimension=collection.dimension,
                metric=collection.metric,
                index_type=collection.config.index_type.value,
                vector_count=len(collection),
                created_at=stats.created_at,
                updated_at=stats.updated_at,
                memory_usage_bytes=stats.memory_usage_bytes,
            ))
        except Exception:
            continue
    
    return CollectionListResponse(
        collections=collections,
        total=total,
    )


@router.get(
    "/{collection_name}",
    response_model=CollectionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
    },
    summary="Get collection info",
    description="Get detailed information about a specific collection.",
)
async def get_collection(
    collection_name: str,
    db: VectorDB = Depends(get_database),
):
    """Get collection information."""
    try:
        collection = db.get_collection(collection_name)
        stats = collection.stats()
        
        return CollectionResponse(
            name=collection.name,
            dimension=collection.dimension,
            metric=collection.metric,
            index_type=collection.config.index_type.value,
            vector_count=len(collection),
            created_at=stats.created_at,
            updated_at=stats.updated_at,
            memory_usage_bytes=stats.memory_usage_bytes,
        )
    
    except CollectionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found"
        )


@router.delete(
    "/{collection_name}",
    response_model=SuccessResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
    },
    summary="Delete a collection",
    description="Delete a collection and all its vectors.",
)
async def delete_collection(
    collection_name: str,
    db: VectorDB = Depends(get_database),
    _auth: bool = Depends(verify_api_key),
):
    """Delete a collection."""
    if not db.delete_collection(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found"
        )
    
    return SuccessResponse(
        message=f"Collection '{collection_name}' deleted successfully"
    )
