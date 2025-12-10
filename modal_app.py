"""
Modal deployment configuration for Chemical Eye API.

Deploy with:
    modal deploy modal_app.py

Run locally with:
    modal serve modal_app.py
"""

import modal

# Create Modal app
app = modal.App("chemeye")

# Define the image with all dependencies and local code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core data
        "earthaccess>=0.9.0",
        "xarray>=2024.1.0",
        "numpy>=1.26.0",
        "h5netcdf>=1.3.0",
        "netCDF4>=1.6.0",
        # API
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        # Database
        "sqlalchemy>=2.0.0",
        # HTTP
        "requests>=2.31.0",
        "httpx>=0.26.0",
        # Utilities
        "python-dotenv>=1.0.0",
    )
    .add_local_dir("src/chemeye", remote_path="/app/chemeye", copy=True)
)

# Create persistent volume for database
volume = modal.Volume.from_name("chemeye-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("chemeye-secrets")],
    allow_concurrent_inputs=100,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI application."""
    import os
    import sys
    
    # Add app to path
    sys.path.insert(0, "/app")
    
    # Set database path to persistent volume
    os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")
    
    from chemeye.api.app import app
    return app


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("chemeye-secrets")],
    timeout=600,
)
def run_methane_detection(
    user_id: str,
    detection_id: str,
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_granules: int = 10,
) -> dict:
    """
    Background worker for heavy methane detection jobs.
    
    Use this for large bbox or long time ranges that might timeout.
    """
    import os
    import sys
    from datetime import datetime
    
    sys.path.insert(0, "/app")
    os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")
    
    from chemeye.database import Detection, DetectionStatus, get_session_maker
    from chemeye.services.methane import MethaneDetector
    
    # Get database session
    SessionLocal = get_session_maker()
    db = SessionLocal()
    
    try:
        # Update status to running
        detection = db.query(Detection).filter(Detection.id == detection_id).first()
        if detection:
            detection.status = DetectionStatus.RUNNING.value
            db.commit()
        
        # Run detection
        detector = MethaneDetector()
        result = detector.detect(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_granules=max_granules,
        )
        
        # Update detection record
        if detection:
            detection.status = DetectionStatus.COMPLETE.value
            detection.result_json = {
                "status": result.status,
                "scanned_count": result.scanned_count,
                "detection_count": len(result.detections),
                "detections": [
                    {
                        "granule_id": d.granule_id,
                        "timestamp": d.timestamp,
                        "lat": d.lat,
                        "lon": d.lon,
                        "plume_size_pixels": d.plume_size_pixels,
                    }
                    for d in result.detections
                ],
            }
            detection.completed_at = datetime.utcnow()
            db.commit()
        
        return {
            "status": result.status,
            "scanned_count": result.scanned_count,
            "detection_count": len(result.detections),
        }
        
    except Exception as e:
        if detection:
            detection.status = DetectionStatus.ERROR.value
            detection.error_message = str(e)
            db.commit()
        raise
    finally:
        db.close()


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("Chemical Eye Modal App")
    print("======================")
    print("\nCommands:")
    print("  modal deploy modal_app.py     # Deploy to Modal")
    print("  modal serve modal_app.py      # Run locally with hot reload")
