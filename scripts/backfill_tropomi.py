"""
Backfill TROPOMI hotspots by looking back up to 7 days to find the most recent
available global granules.

Usage:
    modal run -m scripts.backfill_tropomi
or locally:
    python scripts/backfill_tropomi.py
"""

import os
import sys
from datetime import date, datetime, timedelta

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def store_hotspots(db, day: date, hotspots: list[dict]) -> int:
    from chemeye.database import Detection, DetectionStatus

    count = 0
    for h in hotspots:
        lat = h.get("lat")
        lon = h.get("lon")
        if lat is None or lon is None:
            continue
        bbox_delta = 0.05
        bbox = {
            "min_lon": lon - bbox_delta,
            "min_lat": lat - bbox_delta,
            "max_lon": lon + bbox_delta,
            "max_lat": lat + bbox_delta,
        }
        det = Detection(
            user_id="system",
            detection_type="tropomi_hotspot",
            bbox_json=bbox,
            start_date=day.isoformat(),
            end_date=day.isoformat(),
            status=DetectionStatus.DETECTED.value,
            result_json={
                "granule_ur": h.get("granule_ur", ""),
                "max_ppb": h.get("max_ppb"),
                "mean_ppb": h.get("mean_ppb"),
                "pixel_count": h.get("pixel_count"),
                "timestamp": h.get("timestamp"),
                "plumes": [
                    {
                        "lat": lat,
                        "lon": lon,
                        "z_score": h.get("z_score"),
                        "pixel_count": h.get("pixel_count"),
                    }
                ],
                "max_z_score": h.get("z_score"),
            },
        )
        db.add(det)
        count += 1
    return count


def main():
    from chemeye.database import get_session_maker
    from chemeye.services import tropomi

    Session = get_session_maker()
    db = Session()
    ingested = 0
    try:
        today = datetime.utcnow().date()
        for offset in range(0, 7):
            day = today - timedelta(days=offset)
            hotspots = tropomi.search_daily_hotspots(day)
            if not hotspots:
                print(f"[{day}] no TROPOMI hotspots found")
                continue
            ingested = store_hotspots(db, day, hotspots)
            db.commit()
            print(
                f"[{datetime.utcnow().isoformat()}] ✅ Ingested {ingested} hotspots for {day} "
                f"(stopped after first non-empty day)"
            )
            break
        if ingested == 0:
            print("No hotspots found in last 7 days.")
    finally:
        db.close()


if modal:
    app = modal.App("chemeye-backfill-tropomi")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements("requirements.txt")
        .add_local_dir("src/chemeye", remote_path="/app/chemeye", copy=True)
        .env({"PYTHONPATH": "/root:/root/src:/app", "DATABASE_URL": os.environ["DATABASE_URL"]})
    )
    volume = modal.Volume.from_name("chemeye-data", create_if_missing=True)

    @app.function(image=image, volumes={"/data": volume}, timeout=900)
    def run_backfill():
        main()

    @app.local_entrypoint()
    def entrypoint():
        run_backfill.remote()


if __name__ == "__main__":
    if modal:
        entrypoint()
    else:
        main()
