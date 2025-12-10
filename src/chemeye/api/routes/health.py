"""
Health and status endpoints.
"""

from datetime import datetime

from fastapi import APIRouter

from ... import __version__
from ...config import get_settings
from ..schemas import HealthResponse, StatusResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns basic health status for load balancers and monitoring.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
    )


@router.get("/v1/status", response_model=StatusResponse)
async def api_status():
    """
    API status endpoint.

    Returns detailed status including timestamp.
    """
    settings = get_settings()
    return StatusResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        timestamp=datetime.utcnow(),
    )
