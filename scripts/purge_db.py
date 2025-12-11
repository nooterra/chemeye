"""
Purge fake and stale detections to prepare for a clean backfill.

Usage:
    modal run -m scripts.purge_db
or locally:
    python scripts/purge_db.py
"""

import os
import sys
from datetime import datetime

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    from chemeye.database import Detection, get_session_maker

    Session = get_session_maker()
    db = Session()
    try:
        seed_ids = []
        for det in db.query(Detection).filter(
            Detection.detection_type.in_(["methane_amf", "methane"])
        ):
            gran = None
            if det.result_json:
                gran = det.result_json.get("granule_ur")
            if gran and str(gran).startswith("TEST-SEED"):
                seed_ids.append(det.id)
        seed_deleted = 0
        if seed_ids:
            seed_deleted = (
                db.query(Detection)
                .filter(Detection.id.in_(seed_ids))
                .delete(synchronize_session=False)
            )
        tropomi_deleted = (
            db.query(Detection)
            .filter(Detection.detection_type == "tropomi_hotspot")
            .delete(synchronize_session=False)
        )
        db.commit()
        print(
            f"[{datetime.utcnow().isoformat()}] 🗑️ Purged seed={seed_deleted}, "
            f"tropomi_hotspot={tropomi_deleted}"
        )
    finally:
        db.close()


if modal:
    app = modal.App("chemeye-purge-db")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements("requirements.txt")
        .add_local_dir("src/chemeye", remote_path="/app/chemeye", copy=True)
        .env({"PYTHONPATH": "/root:/root/src:/app", "DATABASE_URL": os.environ["DATABASE_URL"]})
    )
    volume = modal.Volume.from_name("chemeye-data", create_if_missing=True)

    @app.function(image=image, volumes={"/data": volume}, timeout=300)
    def run_purge():
        main()

    @app.local_entrypoint()
    def entrypoint():
        run_purge.remote()


if __name__ == "__main__":
    if modal:
        entrypoint()
    else:
        main()
