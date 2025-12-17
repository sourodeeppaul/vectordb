"""
Custom middleware for the VectorDB server.
"""

from __future__ import annotations

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("vectordb.server")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log request details and timing.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"({duration_ms:.2f}ms)"
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for consistent error handling.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "detail": str(e) if logger.isEnabledFor(logging.DEBUG) else None,
                }
            )


class CORSMiddleware:
    """
    Custom CORS middleware configuration.
    """
    
    @staticmethod
    def get_config():
        from .config import get_config
        config = get_config()
        
        return {
            "allow_origins": config.cors_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }