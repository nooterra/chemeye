"""
Manual trigger to ingest TROPOMI hotspots for the last day.

Usage:
    modal run scripts/ingest_tropomi_now.py
or
    python scripts/ingest_tropomi_now.py  # if running locally with env vars set
"""

import os
import sys
from datetime import datetime

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

# Ensure DB path is set when running under Modal or locally
os.environ.setdefault("DATABASE_URL", "sqlite:////data/chemeye.db")

# Make local source importable when run outside Modal
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    from chemeye.workers import update_global_map

    print(f"[{datetime.utcnow().isoformat()}] Ingesting TROPOMI hotspots...")
    update_global_map(hours=24)
    print("Done.")


# Modal app definition (always present when modal is installed)
if modal:
    app = modal.App("chemeye-tropomi-ingest")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements("requirements.txt")
        .env({"PYTHONPATH": "/root:/root/src", "DATABASE_URL": os.environ["DATABASE_URL"]})
    )

    volume = modal.Volume.from_name("chemeye-data", create_if_missing=True)

    @app.function(image=image, volumes={"/data": volume})
    def run_ingest():
        main()

    @app.local_entrypoint()
    def entrypoint():
        run_ingest.remote()


if __name__ == "__main__":
    if modal:
        entrypoint()
    else:
        main()
