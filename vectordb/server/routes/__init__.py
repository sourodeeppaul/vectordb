"""
API routes for VectorDB server.
"""

from fastapi import APIRouter
from .collections import router as collections_router
from .vectors import router as vectors_router
from .search import router as search_router
from .admin import router as admin_router


def create_api_router() -> APIRouter:
    """Create the main API router with all sub-routers."""
    api_router = APIRouter()
    
    # Include sub-routers
    api_router.include_router(
        collections_router,
        prefix="/collections",
        tags=["Collections"],
    )
    api_router.include_router(
        vectors_router,
        prefix="/collections/{collection_name}/vectors",
        tags=["Vectors"],
    )
    api_router.include_router(
        search_router,
        prefix="/collections/{collection_name}",
        tags=["Search"],
    )
    api_router.include_router(
        admin_router,
        tags=["Admin"],
    )
    
    return api_router
