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
from sqlalchemy import and_

from .database import Detection, DetectionStatus, ProcessedGranule, get_session_maker
from .services.emit import REFLECTANCE_PRODUCT
from .services import tropomi

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


def update_global_map(hours: int = 24, target_day: datetime | None = None) -> None:
    """
    Refresh global hotspots from TROPOMI for the last `hours`.

    - Pull hotspots for today (UTC) and optionally previous hours window.
    - Deduplicate by proximity (~0.05 deg) and day.
    - Mark very old hotspots (>48h) as CLEAR to declutter.
    """
    SessionLocal = get_session_maker()
    db = SessionLocal()

    try:
        # ensure Earthdata auth once
        try:
            earthaccess.login(strategy="netrc")
        except Exception as e:
            logger.warning("Earthaccess login failed: %s", e)

        today = (target_day.date() if isinstance(target_day, datetime) else target_day) or datetime.utcnow().date()
        hotspots = tropomi.search_daily_hotspots(today)

        # optional lookback within the same date range
        if hours > 24:
            earlier = today - timedelta(days=1)
            hotspots += tropomi.search_daily_hotspots(earlier)

        if not hotspots:
            logger.info("No hotspots found for %s", today.isoformat())
            return

        existing = db.query(Detection).filter(Detection.detection_type == "tropomi_hotspot").all()

        new_count = 0
        for h in hotspots:
            lat = h.get("lat")
            lon = h.get("lon")
            if lat is None or lon is None:
                continue

            # Deduplicate: if an existing hotspot within ~0.05 deg and same day, skip
            already = False
            for det in existing:
                bbox = det.bbox_json or {}
                if (
                    bbox
                    and abs((bbox.get("min_lat", 0) + bbox.get("max_lat", 0)) / 2 - lat) < 0.05
                    and abs((bbox.get("min_lon", 0) + bbox.get("max_lon", 0)) / 2 - lon) < 0.05
                    and det.start_date == today.isoformat()
                ):
                    already = True
                    break
            if already:
                continue

            bbox_delta = 0.05  # ~5km
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
                start_date=today.isoformat(),
                end_date=today.isoformat(),
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
                },
            )
            # max_z_score analog
            det.result_json["max_z_score"] = h.get("z_score")
            db.add(det)
            new_count += 1

        # Prune/clear hotspots older than 48h to keep map fresh
        cutoff = datetime.utcnow() - timedelta(hours=48)
        cleared = (
            db.query(Detection)
            .filter(
                Detection.detection_type == "tropomi_hotspot",
                Detection.created_at < cutoff,
                Detection.status == DetectionStatus.DETECTED.value,
            )
            .update({Detection.status: DetectionStatus.CLEAR.value})
        )

        if new_count or cleared:
            db.commit()

        logger.info("TROPOMI update: %d new hotspots, %d cleared", new_count, cleared)
    finally:
        db.close()


def scan_tropomi_daily(hours: int = 6, max_granules: int = 10) -> None:
    """
    Pull recent TROPOMI granules and persist hotspots as detections.
    """
    SessionLocal = get_session_maker()
    db = SessionLocal()

    try:
        granules = tropomi.search_recent_tropomi_granules(hours=hours, count=max_granules)
        if not granules:
            logger.info("No recent TROPOMI granules found")
            return

        existing = {pg.granule_ur for pg in db.query(ProcessedGranule).all()}
        new_count = 0
        det_count = 0

        for g in granules:
            granule_ur = g.get("umm", {}).get("GranuleUR")
            if not granule_ur:
                continue
            if granule_ur in existing:
                continue

            hotspots = tropomi.process_tropomi_granule(g)
            for h in hotspots:
                bbox_delta = 0.05  # ~5km padding
                bbox = {
                    "min_lon": h["lon"] - bbox_delta,
                    "min_lat": h["lat"] - bbox_delta,
                    "max_lon": h["lon"] + bbox_delta,
                    "max_lat": h["lat"] + bbox_delta,
                }
                det = Detection(
                    detection_type="tropomi_hotspot",
                    bbox_json=bbox,
                    start_date=h.get("timestamp", "")[:10],
                    end_date=h.get("timestamp", "")[:10],
                    status=DetectionStatus.DETECTED.value,
                    result_json={
                        "granule_ur": h.get("granule_ur", ""),
                        "max_ppb": h.get("max_ppb"),
                        "mean_ppb": h.get("mean_ppb"),
                        "pixel_count": h.get("pixel_count"),
                        "timestamp": h.get("timestamp"),
                        "plumes": [
                            {
                                "lat": h["lat"],
                                "lon": h["lon"],
                                "z_score": None,
                                "pixel_count": h["pixel_count"],
                            }
                        ],
                        "overlay_image_path": None,
                        "zscore_cog_path": None,
                    },
                )
                # center lat/lon for convenience
                det.lat = h["lat"] if hasattr(det, "lat") else None  # ignored if model lacks columns
                det.lon = h["lon"] if hasattr(det, "lon") else None
                db.add(det)
                det_count += 1

            pg = ProcessedGranule(product=tropomi.TROPOMI_COLLECTION, granule_ur=granule_ur)
            db.add(pg)
            new_count += 1

        if new_count or det_count:
            db.commit()

        logger.info("TROPOMI scan: %d new granules, %d hotspots saved", new_count, det_count)
    finally:
        db.close()
