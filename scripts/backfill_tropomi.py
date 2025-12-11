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
from datetime import datetime, timedelta

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


LOOKBACK_DAYS = 14


def main():
    from chemeye.workers import update_global_map
    from chemeye.services.tropomi import ensure_netrc
    import earthaccess

    try:
        ensure_netrc()
        earthaccess.login(strategy="netrc")
    except Exception as e:
        print(f"⚠️ earthaccess login failed: {e}")

    today = datetime.utcnow().date()
    found = False

    for offset in range(LOOKBACK_DAYS):
        target_day = today - timedelta(days=offset)
        print(f"[{datetime.utcnow().isoformat()}] 🔎 Checking {target_day} for S5P_L2__CH4____HiR...")
        try:
            update_global_map(hours=48, target_day=datetime.combine(target_day, datetime.min.time()))
            found = True
            print(f"✅ Ingest attempted for {target_day}. Continuing to next day to enrich coverage.")
            # continue through window to accumulate multiple days of coverage
        except Exception as e:
            print(f"⚠️  Ingest failed for {target_day}: {e}")

    if not found:
        print("CRITICAL: No TROPOMI data ingested in the last 14 days.")


if modal:
    app = modal.App("chemeye-backfill-tropomi")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements("requirements.txt")
        .add_local_dir("src/chemeye", remote_path="/app/chemeye", copy=True)
        .env({"PYTHONPATH": "/root:/root/src:/app", "DATABASE_URL": os.environ["DATABASE_URL"]})
    )
    volume = modal.Volume.from_name("chemeye-data", create_if_missing=True)

    @app.function(image=image, volumes={"/data": volume}, timeout=1800)
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
