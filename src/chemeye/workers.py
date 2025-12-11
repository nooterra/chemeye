"""
Background workers for automated EMIT scanning and methane detection.

This module is designed to be called from Modal cron functions or other
background execution environments. It scans for recent EMIT L2A granules
and enqueues detection jobs while recording which granules have been
processed.
"""

import logging
from datetime import datetime, timedelta
from typing import Sequence

import earthaccess

from .database import Detection, DetectionStatus, ProcessedGranule, get_session_maker
from .services.emit import REFLECTANCE_PRODUCT

logger = logging.getLogger(__name__)


def get_recent_l2a_granules(hours: int = 24) -> Sequence[dict]:
    """
    Find EMIT L2A reflectance granules whose temporal extent overlaps the
    window [now - hours, now].
    """
    now = datetime.utcnow()
    start = (now - timedelta(hours=hours)).isoformat() + "Z"
    end = now.isoformat() + "Z"

    logger.info("Scanning for recent L2A granules between %s and %s", start, end)

    granules = earthaccess.search_data(
        short_name=REFLECTANCE_PRODUCT,
        temporal=(start, end),
    )

    logger.info("Found %d recent L2A granules", len(granules))
    return granules


def scan_recent_granules(hours: int = 24) -> list[tuple[str, str]]:
    """
    Scan for recent EMIT L2A granules and enqueue new detection jobs.

    Returns:
        List of (granule_ur, detection_id) tuples for newly queued work.
    """
    SessionLocal = get_session_maker()
    db = SessionLocal()

    try:
        granules = get_recent_l2a_granules(hours=hours)
        if not granules:
            logger.info("No recent granules found in the last %d hours", hours)
            return []

        # Build set of existing processed granules
        existing = {
            pg.granule_ur
            for pg in db.query(ProcessedGranule).all()
        }

        jobs: list[tuple[str, str]] = []

        for g in granules:
            granule_ur = g.get("umm", {}).get("GranuleUR")
            if not granule_ur:
                continue

            if granule_ur in existing:
                continue

            # Record processed granule
            pg = ProcessedGranule(
                product=REFLECTANCE_PRODUCT,
                granule_ur=granule_ur,
            )
            db.add(pg)

            # Create a detection job with queued status; parameters can be
            # filled in as we wire richer job orchestration.
            detection = Detection(
                user_id="system",  # placeholder/system actor
                detection_type="methane_amf",
                bbox_json={},
                start_date="",
                end_date="",
                status=DetectionStatus.QUEUED.value,
            )
            db.add(detection)
            db.flush()

            jobs.append((granule_ur, detection.id))

        if jobs:
            db.commit()

        logger.info("Queued %d new granule detection jobs", len(jobs))
        return jobs

    finally:
        db.close()
