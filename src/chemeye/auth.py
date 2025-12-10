"""
API key generation, hashing, and validation.
"""

import hashlib
import secrets
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from .database import APIKey, User


# Key prefix for identification
KEY_PREFIX = "chem_live_"


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash, last4)
    """
    # Generate 32 random bytes, encode as hex
    random_part = secrets.token_hex(24)
    full_key = f"{KEY_PREFIX}{random_part}"

    # Hash for storage (using SHA-256 for simplicity; bcrypt for production)
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    # Last 4 characters for display
    last4 = random_part[-4:]

    return full_key, key_hash, last4


def hash_api_key(key: str) -> str:
    """Hash an API key for comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


def validate_api_key(db: Session, key: str) -> Optional[APIKey]:
    """
    Validate an API key and return the APIKey record if valid.

    Args:
        db: Database session
        key: The full API key string

    Returns:
        APIKey record if valid, None otherwise
    """
    if not key or not key.startswith(KEY_PREFIX):
        return None

    key_hash = hash_api_key(key)

    api_key = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True,  # noqa: E712
    ).first()

    if api_key:
        # Update last used timestamp
        api_key.last_used_at = datetime.utcnow()
        db.commit()

    return api_key


def create_api_key_for_user(
    db: Session,
    user_id: str,
    name: Optional[str] = None,
) -> tuple[str, APIKey]:
    """
    Create a new API key for a user.

    Args:
        db: Database session
        user_id: User ID
        name: Optional label for the key

    Returns:
        Tuple of (full_key, APIKey record)
        NOTE: The full_key is only returned once and cannot be retrieved later!
    """
    full_key, key_hash, last4 = generate_api_key()

    api_key = APIKey(
        user_id=user_id,
        key_hash=key_hash,
        prefix=KEY_PREFIX,
        last4=last4,
        name=name,
    )

    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return full_key, api_key


def create_user(
    db: Session,
    email: str,
    name: Optional[str] = None,
) -> User:
    """
    Create a new user.

    Args:
        db: Database session
        email: User email
        name: Optional user name

    Returns:
        User record
    """
    user = User(email=email, name=name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
