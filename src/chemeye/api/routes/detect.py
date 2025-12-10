"""
Detection endpoints - methane, spectral analysis, etc.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Request

from ...database import Detection, DetectionStatus
from ...services.methane import MethaneDetector
from ..deps import DBSession, RequiredAPIKey, get_request_id
from ..schemas import MethaneDetectRequest, MethaneDetectResponse, MethaneDetectionItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/detect", tags=["Detection"])


@router.post("/methane", response_model=MethaneDetectResponse)
async def detect_methane(
    request: Request,
    body: MethaneDetectRequest,
    api_key: RequiredAPIKey,
    db: DBSession,
):
    """
    Detect methane plumes in the specified region and time range.

    Uses NASA EMIT L2B Methane Plume Complex data to identify confirmed
    methane plumes from satellite hyperspectral imagery.

    **Authentication**: Requires `x-api-key` header.
    """
    request_id = get_request_id(request)

    logger.info(
        f"[{request_id}] Methane detection request: "
        f"bbox={body.bbox.to_tuple()}, dates={body.start_date} to {body.end_date}"
    )

    # Create detection record
    detection = Detection(
        user_id=api_key.user_id,
        detection_type="methane",
        bbox_json={
            "min_lon": body.bbox.min_lon,
            "min_lat": body.bbox.min_lat,
            "max_lon": body.bbox.max_lon,
            "max_lat": body.bbox.max_lat,
        },
        start_date=body.start_date,
        end_date=body.end_date,
        status=DetectionStatus.RUNNING.value,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)

    detection_id = detection.id

    try:
        # Run detection
        detector = MethaneDetector()
        result = detector.detect(
            bbox=body.bbox.to_tuple(),
            start_date=body.start_date,
            end_date=body.end_date,
            max_granules=body.max_granules,
        )

        # Convert detections to response format
        detection_items = [
            MethaneDetectionItem(
                granule_id=d.granule_id,
                timestamp=d.timestamp,
                lat=d.lat,
                lon=d.lon,
                plume_size_pixels=d.plume_size_pixels,
            )
            for d in result.detections
        ]

        # Update detection record
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

        logger.info(
            f"[{request_id}] Methane detection complete: "
            f"status={result.status}, detections={len(result.detections)}"
        )

        return MethaneDetectResponse(
            status=result.status,
            scanned_count=result.scanned_count,
            detection_count=len(result.detections),
            detections=detection_items,
            message=result.message,
            request_id=detection_id,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Methane detection failed: {e}")

        # Update detection record with error
        detection.status = DetectionStatus.ERROR.value
        detection.error_message = str(e)
        db.commit()

        return MethaneDetectResponse(
            status="ERROR",
            scanned_count=0,
            detection_count=0,
            detections=[],
            message=f"Detection failed: {str(e)}",
            request_id=detection_id,
        )
