#!/usr/bin/env python3
"""
CLI tool to create API keys for Chemical Eye.

Usage:
    python -m chemeye.cli.create_key --email user@example.com --name "My Dev Key"
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chemeye.auth import create_api_key_for_user, create_user
from chemeye.database import get_session_maker, init_db, User


def main():
    parser = argparse.ArgumentParser(description="Create an API key for Chemical Eye")
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument("--name", default=None, help="Optional key label")
    args = parser.parse_args()

    # Initialize database
    init_db()

    # Get session
    SessionLocal = get_session_maker()
    db = SessionLocal()

    try:
        # Check if user exists
        user = db.query(User).filter(User.email == args.email).first()

        if not user:
            print(f"📝 Creating new user: {args.email}")
            user = create_user(db, email=args.email)
            print(f"   User ID: {user.id}")
        else:
            print(f"👤 Found existing user: {args.email}")
            print(f"   User ID: {user.id}")

        # Create API key
        print(f"\n🔑 Generating API key...")
        full_key, api_key = create_api_key_for_user(db, user.id, name=args.name)

        print(f"\n{'='*60}")
        print(f"✅ API KEY CREATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"\n   Key:     {full_key}")
        print(f"   Prefix:  {api_key.prefix}")
        print(f"   Last 4:  {api_key.last4}")
        if args.name:
            print(f"   Label:   {args.name}")
        print(f"\n⚠️  SAVE THIS KEY NOW! It cannot be retrieved later.\n")

    finally:
        db.close()


if __name__ == "__main__":
    main()
