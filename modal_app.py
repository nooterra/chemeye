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
        "scikit-learn>=1.4.0",
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
        # Visualization / output
        "matplotlib>=3.8.0",
        "rasterio>=1.3.0",
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


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("chemeye-secrets")],
    timeout=900,
    memory=8192,
    cpu=2.0,
)
def process_granule_job(granule_ur: str, detection_id: str) -> dict:
    """
    Process a single EMIT L2A granule with the AMF methane detector.

    This worker:
      - Loads the specified granule
      - Runs cluster-based AMF with iterative background cleaning
      - Generates an RGB+heatmap overlay image for high-confidence plumes
      - Updates the Detection record with summary metadata
    """
    import os
    import sys
    from datetime import datetime

    sys.path.insert(0, "/app")
    os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")

    from chemeye.database import Detection, DetectionStatus, get_session_maker
    from chemeye.services.emit import REFLECTANCE_PRODUCT, get_emit_service
    from chemeye.services.methane_amf import run_matched_filter, extract_plume_locations
    from chemeye.services.visualization import get_rgb_array, generate_overlay_image, write_z_score_cog

    SessionLocal = get_session_maker()
    db = SessionLocal()

    try:
        detection = db.query(Detection).filter(Detection.id == detection_id).first()
        if not detection:
            raise ValueError(f"Detection {detection_id} not found")

        detection.status = DetectionStatus.RUNNING.value
        db.commit()

        emit_service = get_emit_service()
        emit_service.authenticate()

        import earthaccess

        granules = earthaccess.search_data(
            short_name=REFLECTANCE_PRODUCT,
            granule_name=granule_ur,
            count=1,
        )
        if not granules:
            raise ValueError(f"No L2A granule found for UR {granule_ur}")

        files = emit_service.open_granules([granules[0]])
        if not files:
            raise ValueError(f"Could not open L2A granule {granule_ur}")

        ds_l2a = emit_service.load_dataset(files[0])
        if ds_l2a is None:
            raise ValueError(f"Failed to load L2A dataset for {granule_ur}")

        # Run AMF with cluster-based background and iterative cleaning
        z_map, metadata = run_matched_filter(
            ds_l2a,
            z_threshold=3.0,
            mode="cluster",
            n_clusters=10,
            iterative_background=True,
        )

        max_z = metadata.get("max_z_score", 0.0)

        # Extract plume locations if lat/lon are available
        plumes = []
        lat_array = None
        lon_array = None
        if "lat" in ds_l2a and "lon" in ds_l2a:
            lat_array = ds_l2a["lat"].values
            lon_array = ds_l2a["lon"].values
            plumes = extract_plume_locations(z_map, lat_array, lon_array, min_pixels=5)

        overlay_path = None
        cog_path = None

        # High-confidence threshold for visualization
        visualization_threshold = 4.0

        if max_z >= visualization_threshold:
            rgb = get_rgb_array(ds_l2a)
            overlay_dir = "/data/detections"
            overlay_path = os.path.join(overlay_dir, f"{detection_id}_overlay.png")
            generate_overlay_image(rgb, z_map, output_path=overlay_path, threshold=visualization_threshold)

            cog_path = os.path.join(overlay_dir, f"{detection_id}_zscore.tif")
            if lat_array is not None and lon_array is not None:
                write_z_score_cog(z_map, output_path=cog_path, lon_array=lon_array, lat_array=lat_array)
            else:
                write_z_score_cog(z_map, output_path=cog_path)

        # Update detection record with summary
        status_value = DetectionStatus.CLEAR.value if max_z < visualization_threshold else DetectionStatus.DETECTED.value

        detection.status = status_value
        detection.result_json = {
            "granule_ur": granule_ur,
            "max_z_score": max_z,
            "n_detection_pixels": metadata.get("n_detection_pixels"),
            "mode": metadata.get("mode"),
            "n_clusters": metadata.get("n_clusters"),
            "iterative_background": metadata.get("iterative_background"),
            "plumes": [
                {
                    "lat": p.lat,
                    "lon": p.lon,
                    "z_score": p.z_score,
                    "pixel_count": p.pixel_count,
                    "centroid_row": p.centroid_row,
                    "centroid_col": p.centroid_col,
                }
                for p in plumes
            ],
            "overlay_image_path": overlay_path,
            "zscore_cog_path": cog_path,
        }
        detection.completed_at = datetime.utcnow()
        db.commit()

        # High-confidence alert
        if max_z >= 5.0 and plumes:
            import requests

            webhook_url = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("SLACK_WEBHOOK_URL")
            if webhook_url:
                top = plumes[0]
                content = (
                    "🚨 **Methane Detection**\n"
                    f"**Z-Score:** {max_z:.2f}\n"
                    f"**Loc:** {top.lat:.4f}, {top.lon:.4f}\n"
                    f"**Granule:** {granule_ur}"
                )
                try:
                    requests.post(webhook_url, json={"content": content}, timeout=5)
                except Exception:
                    # Alert failures should not break the worker
                    pass

        return {
            "status": status_value,
            "max_z_score": max_z,
            "granule_ur": granule_ur,
            "n_plumes": len(plumes),
        }

    except Exception as e:
        if "detection" in locals() and detection is not None:
            detection.status = DetectionStatus.ERROR.value
            detection.error_message = str(e)
            db.commit()
        raise
    finally:
        db.close()


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("chemeye-secrets")],
    schedule=modal.Cron("0 * * * *"),
    timeout=600,
)
def cron_scan_recent():
    """
    Hourly cron job to scan for recent EMIT granules and enqueue detection work.
    """
    import os
    import sys

    sys.path.insert(0, "/app")
    os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")

    from chemeye.workers import scan_recent_granules

    jobs = scan_recent_granules(hours=24)
    for granule_ur, detection_id in jobs:
        process_granule_job.spawn(granule_ur, detection_id)


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("Chemical Eye Modal App")
    print("======================")
    print("\nCommands:")
    print("  modal deploy modal_app.py     # Deploy to Modal")
    print("  modal serve modal_app.py      # Run locally with hot reload")
