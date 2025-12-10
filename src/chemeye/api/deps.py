"""
FastAPI dependencies for database sessions and authentication.
"""

import logging
import uuid
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..auth import validate_api_key
from ..database import APIKey, get_db

logger = logging.getLogger(__name__)


def get_request_id(request: Request) -> str:
    """Get or generate a request ID for logging."""
    request_id = request.headers.get("x-request-id")
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    return request_id


async def require_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None,
    db: Session = Depends(get_db),
) -> APIKey:
    """
    Dependency that requires a valid API key.

    Raises 401 if no key provided or invalid.
    """
    if not x_api_key:
        logger.warning("API key missing in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide x-api-key header.",
        )

    api_key = validate_api_key(db, x_api_key)

    if not api_key:
        logger.warning(f"Invalid API key attempted: {x_api_key[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key.",
        )

    logger.info(f"Authenticated: key={api_key.prefix}...{api_key.last4}")
    return api_key


# Type alias for dependency injection
DBSession = Annotated[Session, Depends(get_db)]
RequiredAPIKey = Annotated[APIKey, Depends(require_api_key)]
