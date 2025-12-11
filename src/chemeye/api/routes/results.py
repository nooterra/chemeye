"""
Detection result and visualization endpoints.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ...database import Detection, DetectionStatus
from ..deps import DBSession

router = APIRouter(prefix="/v1/detections", tags=["Detections"])


@router.get("/recent")
async def list_recent_detections(
    request: Request,
    db: DBSession,
) -> list[dict[str, Any]]:
    """
    Return a list of recent methane detection jobs suitable for the dashboard.

    This focuses on AMF-based background jobs (detection_type='methane_amf')
    and returns a flattened view combining Detection and result_json fields.
    """
    base_url = str(request.base_url).rstrip("/")

    q = db.query(Detection).order_by(Detection.created_at.desc()).limit(200)

    items: list[dict[str, Any]] = []

    for det in q.all():
        result = det.result_json or {}

        status_upper = det.status.upper() if det.status else "UNKNOWN"
        granule_ur = result.get("granule_ur", "")
        max_z = float(result.get("max_z_score", 0.0))

        # Choose representative lat/lon: first plume if present
        lat = None
        lon = None
        plumes = result.get("plumes") or []
        if plumes:
            first = plumes[0]
            lat = float(first.get("lat"))
            lon = float(first.get("lon"))

        # Fallback to bbox center if no plumes but bbox_json exists
        if (lat is None or lon is None) and det.bbox_json:
            bbox = det.bbox_json
            min_lon = bbox.get("min_lon")
            min_lat = bbox.get("min_lat")
            max_lon = bbox.get("max_lon")
            max_lat = bbox.get("max_lat")
            if None not in (min_lon, min_lat, max_lon, max_lat):
                lon = (min_lon + max_lon) / 2.0
                lat = (min_lat + max_lat) / 2.0

        # Bounds for bitmap layer: small box around lat/lon if no better info
        if lat is not None and lon is not None:
            delta = 0.02  # ~2 km at mid-latitudes, just for visualization
            bounds = [lon - delta, lat - delta, lon + delta, lat + delta]
        else:
            bounds = [0.0, 0.0, 0.0, 0.0]

        overlay_url = f"{base_url}/v1/detections/{det.id}/overlay"

        items.append(
            {
                "id": det.id,
                "granule_ur": granule_ur,
                "status": status_upper,
                "detection_type": det.detection_type,
                "max_z_score": max_z,
                "timestamp": (det.completed_at or det.created_at).isoformat() + "Z",
                "lat": lat,
                "lon": lon,
                "overlay_url": overlay_url,
                "bounds": bounds,
            }
        )

    return items


@router.get("/{detection_id}/overlay")
async def get_detection_overlay(
    detection_id: str,
    db: DBSession,
):
    """
    Return the overlay PNG for a completed detection, if available.
    """
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    if detection.status not in {
        DetectionStatus.COMPLETE.value,
        DetectionStatus.CLEAR.value,
        DetectionStatus.DETECTED.value,
    }:
        raise HTTPException(status_code=409, detail="Detection not complete")

    # Prefer path recorded in result_json, fall back to convention
    result = detection.result_json or {}
    overlay_path_str = result.get("overlay_image_path")

    if overlay_path_str:
        overlay_path = Path(overlay_path_str)
    else:
        overlay_path = Path("/data/detections") / f"{detection_id}_overlay.png"

    if not overlay_path.is_file():
        raise HTTPException(status_code=404, detail="Overlay image not available")

    return FileResponse(
        path=overlay_path,
        media_type="image/png",
        filename=overlay_path.name,
    )
