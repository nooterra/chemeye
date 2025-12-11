"""
Seed the live database with a synthetic high-confidence detection so the
dashboard has something to render immediately.

Usage:
    modal run scripts/seed_live_db.py

This uses DATABASE_URL from the environment (Modal injects this for the live
volume) and inserts a Detection with status DETECTED plus a fake plume.
"""

import os
import uuid
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from chemeye.database import Base, Detection, DetectionStatus


def seed():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")

    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
    )
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        Base.metadata.create_all(engine)

        detection_id = str(uuid.uuid4())

        # Permian Basin test plume
        lat = 31.85
        lon = -103.10
        bbox = {
            "min_lon": lon - 0.05,
            "min_lat": lat - 0.05,
            "max_lon": lon + 0.05,
            "max_lat": lat + 0.05,
        }

        detection = Detection(
            id=detection_id,
            user_id="system",
            detection_type="methane_amf",
            bbox_json=bbox,
            start_date="2025-01-01",
            end_date="2025-01-01",
            status=DetectionStatus.DETECTED.value,
            result_json={
                "granule_ur": "TEST-SEED-PERMIAN-001",
                "max_z_score": 8.5,
                "n_detection_pixels": 250,
                "mode": "cluster",
                "n_clusters": 5,
                "iterative_background": True,
                "plumes": [
                    {
                        "lat": lat,
                        "lon": lon,
                        "z_score": 8.5,
                        "pixel_count": 250,
                        "centroid_row": 100,
                        "centroid_col": 120,
                    }
                ],
                # No real overlay, but the frontend will still render the point.
                "overlay_image_path": None,
                "zscore_cog_path": None,
            },
            completed_at=datetime.utcnow(),
        )

        db.add(detection)
        db.commit()
        print(f"✅ Inserted Detection ID: {detection_id}")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
