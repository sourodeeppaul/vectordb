"""
Vector CRUD endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List
import numpy as np

from ..models import (
    AddVectorRequest,
    AddVectorsRequest,
    AddVectorsResponse,
    VectorResponse,
    UpdateVectorRequest,
    DeleteVectorsRequest,
    DeleteVectorsResponse,
    SuccessResponse,
    ErrorResponse,
)
from ..dependencies import get_database, get_collection, verify_api_key
from ...core.database import VectorDB
from ...core.collection import Collection
from ...core.exceptions import VectorNotFoundError, VectorExistsError

router = APIRouter()


@router.post(
    "",
    response_model=AddVectorsResponse,
    status_code=201,
    summary="Add vectors",
    description="Add one or more vectors to the collection.",
)
async def add_vectors(
    collection_name: str,
    request: AddVectorsRequest,
    collection: Collection = Depends(get_collection),
    _auth: bool = Depends(verify_api_key),
):
    """Add vectors to a collection."""
    added = 0
    failed = 0
    errors = []
    
    for vec_data in request.vectors:
        try:
            if len(vec_data.vector) != collection.dimension:
                raise ValueError(
                    f"Vector dimension {len(vec_data.vector)} != "
                    f"collection dimension {collection.dimension}"
                )
            
            vector = np.array(vec_data.vector, dtype=np.float32)
            collection.add(vec_data.id, vector, vec_data.metadata)
            added += 1
        
        except VectorExistsError:
            failed += 1
            errors.append({
                "id": vec_data.id,
                "error": f"Vector '{vec_data.id}' already exists"
            })
        except Exception as e:
            failed += 1
            errors.append({
                "id": vec_data.id,
                "error": str(e)
            })
    
    return AddVectorsResponse(
        added_count=added,
        failed_count=failed,
        errors=errors if errors else None,
    )


@router.post(
    "/single",
    response_model=SuccessResponse,
    status_code=201,
    summary="Add single vector",
    description="Add a single vector to the collection.",
)
async def add_single_vector(
    collection_name: str,
    request: AddVectorRequest,
    collection: Collection = Depends(get_collection),
    _auth: bool = Depends(verify_api_key),
):
    """Add a single vector."""
    try:
        if len(request.vector) != collection.dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension {len(request.vector)} != "
                       f"collection dimension {collection.dimension}"
            )
        
        vector = np.array(request.vector, dtype=np.float32)
        collection.add(request.id, vector, request.metadata)
        
        return SuccessResponse(
            message=f"Vector '{request.id}' added successfully",
            data={"id": request.id}
        )
    
    except VectorExistsError:
        raise HTTPException(
            status_code=409,
            detail=f"Vector '{request.id}' already exists"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/{vector_id}",
    response_model=VectorResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Vector not found"},
    },
    summary="Get vector",
    description="Get a vector by ID.",
)
async def get_vector(
    collection_name: str,
    vector_id: str = Path(..., min_length=1),
    include_vector: bool = Query(default=True),
    collection: Collection = Depends(get_collection),
):
    """Get a vector by ID."""
    result = collection.get(vector_id, include_vector=include_vector)
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Vector '{vector_id}' not found"
        )
    
    return VectorResponse(
        id=vector_id,
        vector=result.get("vector", []).tolist() if include_vector and "vector" in result else None,
        metadata=result.get("metadata"),
    )


@router.put(
    "/{vector_id}",
    response_model=SuccessResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Vector not found"},
    },
    summary="Update vector",
    description="Update a vector's data or metadata.",
)
async def update_vector(
    collection_name: str,
    vector_id: str,
    request: UpdateVectorRequest,
    collection: Collection = Depends(get_collection),
    _auth: bool = Depends(verify_api_key),
):
    """Update a vector."""
    if request.vector is None and request.metadata is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'vector' or 'metadata' must be provided"
        )
    
    try:
        vector = None
        if request.vector is not None:
            if len(request.vector) != collection.dimension:
                raise HTTPException(
                    status_code=400,
                    detail="Vector dimension mismatch"
                )
            vector = np.array(request.vector, dtype=np.float32)
        
        collection.update(vector_id, vector=vector, metadata=request.metadata)
        
        return SuccessResponse(
            message=f"Vector '{vector_id}' updated successfully"
        )
    
    except VectorNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Vector '{vector_id}' not found"
        )


@router.delete(
    "/{vector_id}",
    response_model=SuccessResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Vector not found"},
    },
    summary="Delete vector",
    description="Delete a vector by ID.",
)
async def delete_vector(
    collection_name: str,
    vector_id: str,
    collection: Collection = Depends(get_collection),
    _auth: bool = Depends(verify_api_key),
):
    """Delete a vector."""
    if not collection.delete(vector_id):
        raise HTTPException(
            status_code=404,
            detail=f"Vector '{vector_id}' not found"
        )
    
    return SuccessResponse(
        message=f"Vector '{vector_id}' deleted successfully"
    )


@router.post(
    "/delete",
    response_model=DeleteVectorsResponse,
    summary="Delete multiple vectors",
    description="Delete multiple vectors by their IDs.",
)
async def delete_vectors(
    collection_name: str,
    request: DeleteVectorsRequest,
    collection: Collection = Depends(get_collection),
    _auth: bool = Depends(verify_api_key),
):
    """Delete multiple vectors."""
    result = collection.delete_many(request.ids)
    
    return DeleteVectorsResponse(
        deleted_count=result["deleted"],
        not_found_count=result["not_found"],
    )


@router.get(
    "",
    response_model=List[VectorResponse],
    summary="List vectors",
    description="List vectors in the collection with pagination.",
)
async def list_vectors(
    collection_name: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    include_vector: bool = Query(default=False),
    collection: Collection = Depends(get_collection),
):
    """List vectors with pagination."""
    ids = collection.list_ids(limit=limit, offset=offset)
    
    results = []
    for id in ids:
        data = collection.get(id, include_vector=include_vector)
        if data:
            results.append(VectorResponse(
                id=id,
                vector=data.get("vector", []).tolist() if include_vector and "vector" in data else None,
                metadata=data.get("metadata"),
            ))
    
    return results
