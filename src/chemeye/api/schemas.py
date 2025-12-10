"""
Pydantic request/response schemas for the API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# --- Request Schemas ---


class BoundingBox(BaseModel):
    """Geographic bounding box."""

    min_lon: float = Field(..., ge=-180, le=180, description="Minimum longitude")
    min_lat: float = Field(..., ge=-90, le=90, description="Minimum latitude")
    max_lon: float = Field(..., ge=-180, le=180, description="Maximum longitude")
    max_lat: float = Field(..., ge=-90, le=90, description="Maximum latitude")

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


class MethaneDetectRequest(BaseModel):
    """Request for methane detection."""

    bbox: BoundingBox = Field(..., description="Geographic bounding box")
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="End date (YYYY-MM-DD)")
    max_granules: int = Field(default=10, ge=1, le=50, description="Max granules to scan")

    class Config:
        json_schema_extra = {
            "example": {
                "bbox": {
                    "min_lon": -117.5,
                    "min_lat": 35.2,
                    "max_lon": -117.0,
                    "max_lat": 35.7,
                },
                "start_date": "2023-05-01",
                "end_date": "2023-08-30",
                "max_granules": 10,
            }
        }


# --- Response Schemas ---


class MethaneDetectionItem(BaseModel):
    """Single methane plume detection."""

    granule_id: str
    timestamp: str
    lat: float
    lon: float
    plume_size_pixels: int


class MethaneDetectResponse(BaseModel):
    """Response for methane detection."""

    status: str = Field(..., description="DETECTED, CLEAR, EMPTY, or ERROR")
    scanned_count: int = Field(..., description="Number of granules scanned")
    detection_count: int = Field(default=0, description="Number of plumes detected")
    detections: list[MethaneDetectionItem] = Field(default_factory=list)
    message: Optional[str] = Field(default=None, description="Additional info or error message")
    request_id: Optional[str] = Field(default=None, description="Detection request ID for tracking")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    environment: str


class StatusResponse(BaseModel):
    """API status response."""

    status: str = "ok"
    version: str
    environment: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


# --- API Key Schemas ---


class APIKeyCreate(BaseModel):
    """Request to create a new API key."""

    email: str = Field(..., description="User email")
    name: Optional[str] = Field(default=None, description="Optional key label")


class APIKeyResponse(BaseModel):
    """Response after creating an API key."""

    key: str = Field(..., description="The API key (only shown once!)")
    prefix: str
    last4: str
    name: Optional[str]
    created_at: datetime
    message: str = "Save this key securely. It cannot be retrieved again."
