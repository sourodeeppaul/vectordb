"""
Search endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
import numpy as np
import time

from ..models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    BatchSearchRequest,
    BatchSearchResponse,
    ErrorResponse,
)
from ..dependencies import get_collection
from ...core.collection import Collection

router = APIRouter()


def build_filter_fn(search_filter):
    """Build a filter function from search filter."""
    if search_filter is None:
        return None
    
    def filter_fn(id: str, metadata: dict) -> bool:
        if search_filter.match:
            for key, value in search_filter.match.items():
                if metadata.get(key) != value:
                    return False
        
        if search_filter.conditions:
            for condition in search_filter.conditions:
                field_value = metadata.get(condition.field)
                
                if condition.operator == "eq":
                    if field_value != condition.value:
                        return False
                elif condition.operator == "ne":
                    if field_value == condition.value:
                        return False
                elif condition.operator == "gt":
                    if field_value is None or field_value <= condition.value:
                        return False
                elif condition.operator == "gte":
                    if field_value is None or field_value < condition.value:
                        return False
                elif condition.operator == "lt":
                    if field_value is None or field_value >= condition.value:
                        return False
                elif condition.operator == "lte":
                    if field_value is None or field_value > condition.value:
                        return False
                elif condition.operator == "in":
                    if field_value not in condition.value:
                        return False
                elif condition.operator == "nin":
                    if field_value in condition.value:
                        return False
        
        return True
    
    return filter_fn


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search vectors",
    description="Search for similar vectors using k-nearest neighbors.",
)
async def search_vectors(
    collection_name: str,
    request: SearchRequest,
    collection: Collection = Depends(get_collection),
):
    """Search for similar vectors."""
    if len(request.vector) != collection.dimension:
        raise HTTPException(
            status_code=400,
            detail=f"Query dimension {len(request.vector)} != "
                   f"collection dimension {collection.dimension}"
        )
    
    query = np.array(request.vector, dtype=np.float32)
    
    filter_dict = None
    if request.filter and request.filter.match:
        filter_dict = request.filter.match
    
    start_time = time.time()
    
    results = collection.search(
        query=query,
        k=request.k,
        filter=filter_dict,
        include_vector=request.include_vector,
        include_metadata=request.include_metadata,
    )
    
    search_time_ms = (time.time() - start_time) * 1000
    
    search_results = []
    for r in results:
        search_results.append(SearchResult(
            id=r.id,
            distance=r.distance,
            score=r.score,
            vector=r.vector.tolist() if r.vector is not None else None,
            metadata=r.metadata,
        ))
    
    return SearchResponse(
        results=search_results,
        total=len(search_results),
        search_time_ms=search_time_ms,
    )


@router.post(
    "/search/batch",
    response_model=BatchSearchResponse,
    summary="Batch search",
    description="Search with multiple query vectors.",
)
async def batch_search(
    collection_name: str,
    request: BatchSearchRequest,
    collection: Collection = Depends(get_collection),
):
    """Batch search with multiple queries."""
    for i, vec in enumerate(request.vectors):
        if len(vec) != collection.dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Query {i} dimension {len(vec)} != "
                       f"collection dimension {collection.dimension}"
            )
    
    queries = np.array(request.vectors, dtype=np.float32)
    
    filter_dict = None
    if request.filter and request.filter.match:
        filter_dict = request.filter.match
    
    start_time = time.time()
    
    all_results = collection.search_batch(
        queries=queries,
        k=request.k,
        filter=filter_dict,
        include_vector=request.include_vector,
    )
    
    search_time_ms = (time.time() - start_time) * 1000
    
    batch_results = []
    for results in all_results:
        query_results = []
        for r in results:
            query_results.append(SearchResult(
                id=r.id,
                distance=r.distance,
                score=r.score,
                vector=r.vector.tolist() if r.vector is not None else None,
                metadata=r.metadata,
            ))
        batch_results.append(query_results)
    
    return BatchSearchResponse(
        results=batch_results,
        total_queries=len(request.vectors),
        search_time_ms=search_time_ms,
    )


@router.post(
    "/query",
    response_model=SearchResponse,
    summary="Query by metadata",
    description="Query vectors by metadata without similarity search.",
)
async def query_by_metadata(
    collection_name: str,
    filter: dict,
    limit: int = 100,
    include_vector: bool = False,
    collection: Collection = Depends(get_collection),
):
    """Query vectors by metadata."""
    results = collection.query_by_metadata(
        filter=filter,
        limit=limit,
        include_vector=include_vector,
    )
    
    search_results = []
    for r in results:
        search_results.append(SearchResult(
            id=r["id"],
            distance=0.0,
            score=1.0,
            vector=r.get("vector", []).tolist() if include_vector else None,
            metadata=r.get("metadata"),
        ))
    
    return SearchResponse(
        results=search_results,
        total=len(search_results),
        search_time_ms=0,
    )
