"""
API Routes for RLM.
"""

from fastapi import APIRouter
from .query_route import router as query_router

api_router = APIRouter()

api_router.include_router(query_router, prefix="/rlm", tags=["RLM Query"])

router = api_router

__all__ = ["router", "api_router"]
