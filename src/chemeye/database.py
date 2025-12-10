"""
Database models and session management.
Uses SQLAlchemy with SQLite (Postgres-ready via connection string swap).
"""

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class DetectionStatus(str, Enum):
    """Status of a detection request."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


class User(Base):
    """User account."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    api_keys: Mapped[list["APIKey"]] = relationship("APIKey", back_populates="user")
    detections: Mapped[list["Detection"]] = relationship("Detection", back_populates="user")


class APIKey(Base):
    """API key for authentication."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    key_hash: Mapped[str] = mapped_column(String(128))  # bcrypt hash
    prefix: Mapped[str] = mapped_column(String(20))  # e.g., "chem_live_"
    last4: Mapped[str] = mapped_column(String(4))  # Last 4 chars for display
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Optional label
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys")


class Detection(Base):
    """Detection request and result."""

    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)

    # Request details
    detection_type: Mapped[str] = mapped_column(String(50), index=True)  # methane, spectral, nitrogen
    bbox_json: Mapped[dict] = mapped_column(JSON)  # {min_lon, min_lat, max_lon, max_lat}
    start_date: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD
    end_date: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD

    # Status tracking
    status: Mapped[str] = mapped_column(String(20), default=DetectionStatus.PENDING.value)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Results
    result_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="detections")


# Database engine and session
def get_engine():
    """Create database engine."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    )


def get_session_maker():
    """Create session maker."""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    settings = get_settings()
    
    # Ensure data directory exists for SQLite
    if "sqlite" in settings.database_url:
        import re
        match = re.search(r"sqlite:///(.+)", settings.database_url)
        if match:
            db_path = Path(match.group(1))
            db_path.parent.mkdir(parents=True, exist_ok=True)
    
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session."""
    SessionLocal = get_session_maker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
