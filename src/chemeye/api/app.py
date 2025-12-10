"""
Main FastAPI application.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import get_settings
from ..database import init_db
from .routes import detect, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting Chemical Eye API v{__version__}")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down Chemical Eye API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Chemical Eye API",
        description=(
            "Planetary Chemical Intelligence from Orbit.\n\n"
            "Chemical Eye provides hyperspectral analysis for detecting "
            "chemicals, minerals, and environmental signatures from satellite imagery."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and logging middleware
    @app.middleware("http")
    async def add_request_id_and_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
        start_time = time.time()

        # Add request ID to state for access in routes
        request.state.request_id = request_id

        response = await call_next(request)

        duration = time.time() - start_time
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"-> {response.status_code} ({duration:.3f}s)"
        )

        response.headers["x-request-id"] = request_id
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.is_dev else None,
                "request_id": request_id,
            },
        )

    # Include routers
    app.include_router(health.router)
    app.include_router(detect.router)

    return app


# Create app instance
app = create_app()
