"""
Admin script to create API keys in the PRODUCTION Modal environment.

Usage:
    modal run admin.py
"""

import modal

# Create a new app for admin tasks
app = modal.App("chemeye-admin")

# Get the production volume (same one used by main app)
volume = modal.Volume.from_name("chemeye-data")

# Use the same image as the main app
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sqlalchemy>=2.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
    )
    .add_local_dir("src/chemeye", remote_path="/app/chemeye", copy=True)
)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("chemeye-secrets")],
)
def create_prod_key(email: str = "aiden@chemeye.io"):
    """Create an API key in the production database."""
    import os
    import sys
    import secrets as py_secrets
    import hashlib
    import uuid
    
    # Add app to path
    sys.path.insert(0, "/app")
    os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")
    
    print(f"🔑 Generating Production Key for {email}...")
    
    # Direct database connection
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from chemeye.database import Base, User, APIKey
    
    engine = create_engine("sqlite:////data/chemeye.db")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if user exists
        user = session.query(User).filter(User.email == email).first()
        
        if not user:
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                name="Admin",
            )
            session.add(user)
            session.flush()
            print(f"   Created user: {email}")
        else:
            print(f"   Found existing user: {email}")
        
        # Generate key
        key_prefix = "chem_live_"
        random_part = py_secrets.token_hex(24)
        full_key = f"{key_prefix}{random_part}"
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        last4 = random_part[-4:]
        
        # Create API key
        api_key = APIKey(
            id=str(uuid.uuid4()),
            user_id=user.id,
            key_hash=key_hash,
            prefix=key_prefix,
            last4=last4,
            name="Production Master Key",
            is_active=True,
        )
        session.add(api_key)
        session.commit()
        
        # Commit the volume to persist changes
        volume.commit()
        
        print(f"\n{'='*60}")
        print(f"✅ PRODUCTION API KEY CREATED")
        print(f"{'='*60}")
        print(f"\n   Key: {full_key}")
        print(f"   Prefix: {key_prefix}")
        print(f"   Last 4: {last4}")
        print(f"\n⚠️  SAVE THIS KEY - it cannot be retrieved later!\n")
        
        return full_key
        
    finally:
        session.close()


@app.local_entrypoint()
def main():
    """Run the admin task."""
    key = create_prod_key.remote("aiden@chemeye.io")
    print(f"\n🎉 Production key created: {key}")
