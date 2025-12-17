"""
Pydantic models for API requests and responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import time


# =============================================================================
# ENUMS
# =============================================================================

class MetricType(str, Enum):
    """Distance metric types."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT = "dot"
    MANHATTAN = "manhattan"


class IndexType(str, Enum):
    """Index types."""
    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"


# =============================================================================
# COMMON MODELS
# =============================================================================

class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)


# =============================================================================
# COLLECTION MODELS
# =============================================================================

class CreateCollectionRequest(BaseModel):
    """Request to create a new collection."""
    name: str = Field(..., min_length=1, max_length=256, pattern=r'^[a-zA-Z0-9_\-]+$')
    dimension: int = Field(..., ge=1, le=65536)
    metric: MetricType = MetricType.EUCLIDEAN
    index_type: IndexType = IndexType.FLAT
    
    # Optional settings
    normalize: bool = False
    description: Optional[str] = None
    
    # Index-specific config
    hnsw_config: Optional[Dict[str, Any]] = None
    ivf_config: Optional[Dict[str, Any]] = None
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "documents",
                    "dimension": 384,
                    "metric": "cosine",
                    "index_type": "hnsw",
                    "description": "Document embeddings"
                }
            ]
        }
    }


class CollectionResponse(BaseModel):
    """Collection information response."""
    name: str
    dimension: int
    metric: str
    index_type: str
    vector_count: int
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    description: Optional[str] = None
    
    # Stats
    memory_usage_bytes: Optional[int] = None
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "documents",
                    "dimension": 384,
                    "metric": "cosine",
                    "index_type": "hnsw",
                    "vector_count": 10000,
                    "memory_usage_bytes": 5242880
                }
            ]
        }
    }


class CollectionListResponse(BaseModel):
    """List of collections response."""
    collections: List[CollectionResponse]
    total: int


# =============================================================================
# VECTOR MODELS
# =============================================================================

class VectorData(BaseModel):
    """Single vector data."""
    id: str = Field(..., min_length=1, max_length=256)
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('vector')
    @classmethod
    def validate_vector(cls, v):
        if not v:
            raise ValueError("Vector cannot be empty")
        if len(v) > 65536:
            raise ValueError("Vector dimension too large")
        return v


class AddVectorRequest(BaseModel):
    """Request to add a single vector."""
    id: str = Field(..., min_length=1, max_length=256)
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "doc_001",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "metadata": {"title": "Example", "category": "demo"}
                }
            ]
        }
    }


class AddVectorsRequest(BaseModel):
    """Request to add multiple vectors."""
    vectors: List[VectorData]
    
    @field_validator('vectors')
    @classmethod
    def validate_vectors(cls, v):
        if not v:
            raise ValueError("Vectors list cannot be empty")
        if len(v) > 10000:
            raise ValueError("Too many vectors in single request (max 10000)")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "vectors": [
                        {"id": "doc_001", "vector": [0.1, 0.2, 0.3], "metadata": {"title": "Doc 1"}},
                        {"id": "doc_002", "vector": [0.4, 0.5, 0.6], "metadata": {"title": "Doc 2"}}
                    ]
                }
            ]
        }
    }


class AddVectorsResponse(BaseModel):
    """Response for batch vector addition."""
    success: bool = True
    added_count: int
    failed_count: int = 0
    errors: Optional[List[Dict[str, str]]] = None


class VectorResponse(BaseModel):
    """Single vector response."""
    id: str
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Include vector or not
    include_vector: bool = False


class UpdateVectorRequest(BaseModel):
    """Request to update a vector."""
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('vector', 'metadata')
    @classmethod
    def at_least_one(cls, v, info):
        return v


class DeleteVectorsRequest(BaseModel):
    """Request to delete multiple vectors."""
    ids: List[str]
    
    @field_validator('ids')
    @classmethod
    def validate_ids(cls, v):
        if not v:
            raise ValueError("IDs list cannot be empty")
        if len(v) > 10000:
            raise ValueError("Too many IDs (max 10000)")
        return v


class DeleteVectorsResponse(BaseModel):
    """Response for batch deletion."""
    success: bool = True
    deleted_count: int
    not_found_count: int = 0


# =============================================================================
# SEARCH MODELS
# =============================================================================

class FilterCondition(BaseModel):
    """Filter condition for metadata filtering."""
    field: str
    operator: str = Field(default="eq", pattern=r'^(eq|ne|gt|gte|lt|lte|in|nin)$')
    value: Any


class SearchFilter(BaseModel):
    """Search filter with multiple conditions."""
    conditions: Optional[List[FilterCondition]] = None
    # Simple key-value filter (equality)
    match: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Vector search request."""
    vector: List[float]
    k: int = Field(default=10, ge=1, le=10000)
    
    # Filtering
    filter: Optional[SearchFilter] = None
    
    # What to include in results
    include_vector: bool = False
    include_metadata: bool = True
    
    # Index-specific params
    ef: Optional[int] = None       # HNSW ef_search
    n_probe: Optional[int] = None  # IVF n_probe
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "k": 10,
                    "filter": {"match": {"category": "documents"}},
                    "include_metadata": True
                }
            ]
        }
    }


class SearchResult(BaseModel):
    """Single search result."""
    id: str
    distance: float
    score: float
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    total: int
    search_time_ms: float


class BatchSearchRequest(BaseModel):
    """Batch search request."""
    vectors: List[List[float]]
    k: int = Field(default=10, ge=1, le=1000)
    filter: Optional[SearchFilter] = None
    include_vector: bool = False
    include_metadata: bool = True
    
    @field_validator('vectors')
    @classmethod
    def validate_vectors(cls, v):
        if not v:
            raise ValueError("Vectors list cannot be empty")
        if len(v) > 100:
            raise ValueError("Too many queries (max 100)")
        return v


class BatchSearchResponse(BaseModel):
    """Batch search response."""
    results: List[List[SearchResult]]
    total_queries: int
    search_time_ms: float


# =============================================================================
# ADMIN MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    uptime_seconds: float


class DatabaseInfoResponse(BaseModel):
    """Database information response."""
    version: str
    collection_count: int
    total_vectors: int
    total_memory_mb: float
    data_dir: Optional[str] = None
    uptime_seconds: float
    collections: Dict[str, Dict[str, Any]]


class SaveRequest(BaseModel):
    """Save database request."""
    path: Optional[str] = None


class SaveResponse(BaseModel):
    """Save database response."""
    success: bool = True
    path: str
    collections_saved: int