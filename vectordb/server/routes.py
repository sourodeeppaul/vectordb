"""
Combined API routes for VectorDB server.

This module provides the main router that combines all
sub-routes for collections, vectors, search, and admin endpoints.
"""

from fastapi import APIRouter

# Import route modules from routes subpackage
from .routes.collections import router as collections_router
from .routes.vectors import router as vectors_router
from .routes.search import router as search_router
from .routes.admin import router as admin_router

# Create main router
router = APIRouter()

# Include sub-routers with prefixes
router.include_router(
    collections_router,
    prefix="/collections",
    tags=["collections"],
)

router.include_router(
    vectors_router,
    prefix="/vectors",
    tags=["vectors"],
)

router.include_router(
    search_router,
    prefix="/search",
    tags=["search"],
)

router.include_router(
    admin_router,
    prefix="/admin",
    tags=["admin"],
)


__all__ = ["router"]
