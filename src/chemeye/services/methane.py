"""
Methane plume detection from EMIT L2B data.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import xarray as xr

from .emit import get_emit_service

logger = logging.getLogger(__name__)


@dataclass
class MethaneDetection:
    """Single methane plume detection."""

    granule_id: str
    timestamp: str
    lat: float
    lon: float
    plume_size_pixels: int


@dataclass
class MethaneResult:
    """Result of methane detection scan."""

    status: str  # "DETECTED", "CLEAR", "EMPTY", "ERROR"
    scanned_count: int
    detections: list[MethaneDetection]
    message: Optional[str] = None


class MethaneDetector:
    """Detector for methane plumes using EMIT L2B data."""

    def __init__(self):
        self.emit_service = get_emit_service()

    def detect(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_granules: int = 10,
    ) -> MethaneResult:
        """
        Detect methane plumes in the specified region and time range.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            max_granules: Maximum number of granules to scan

        Returns:
            MethaneResult with detection details
        """
        logger.info(f"Starting methane detection: bbox={bbox}, dates={start_date} to {end_date}")

        # Search for granules
        try:
            granules = self.emit_service.search_methane_granules(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                count=max_granules,
            )
        except Exception as e:
            logger.error(f"Granule search failed: {e}")
            return MethaneResult(
                status="ERROR",
                scanned_count=0,
                detections=[],
                message=f"Search failed: {str(e)}",
            )

        if not granules:
            logger.info("No EMIT passes found over this location/time")
            return MethaneResult(
                status="EMPTY",
                scanned_count=0,
                detections=[],
                message="No EMIT passes found over this location/time.",
            )

        logger.info(f"Found {len(granules)} granules, scanning for plumes...")

        # Open granules for streaming
        try:
            files = self.emit_service.open_granules(granules)
        except Exception as e:
            logger.error(f"Failed to open granules: {e}")
            return MethaneResult(
                status="ERROR",
                scanned_count=0,
                detections=[],
                message=f"Failed to open data: {str(e)}",
            )

        detections: list[MethaneDetection] = []

        for i, file_obj in enumerate(files):
            try:
                detection = self._scan_granule(file_obj, granules[i])
                if detection:
                    detections.append(detection)
                    logger.info(
                        f"DETECTED: Plume in granule {i} | "
                        f"Size: {detection.plume_size_pixels} pixels | "
                        f"Location: ({detection.lat:.4f}, {detection.lon:.4f})"
                    )
            except Exception as e:
                logger.warning(f"Error scanning granule {i}: {e}")
                continue

        if not detections:
            logger.info("Clean scan - no plumes detected")
            return MethaneResult(
                status="CLEAR",
                scanned_count=len(granules),
                detections=[],
            )

        logger.info(f"Detection complete: {len(detections)} plumes found")
        return MethaneResult(
            status="DETECTED",
            scanned_count=len(granules),
            detections=detections,
        )

    def _scan_granule(self, file_obj, granule_meta: dict) -> Optional[MethaneDetection]:
        """
        Scan a single granule for methane plumes.

        Args:
            file_obj: File object from earthaccess.open()
            granule_meta: Granule metadata dict

        Returns:
            MethaneDetection if plume found, None otherwise
        """
        ds = self.emit_service.load_dataset(file_obj)
        if ds is None:
            return None

        try:
            # Check for plume mask variable
            if "ch4_plume_complex" not in ds.variables:
                logger.debug("No ch4_plume_complex variable in granule")
                ds.close()
                return None

            # Sum plume pixels
            plume_sum = int(ds["ch4_plume_complex"].sum().compute())

            if plume_sum == 0:
                ds.close()
                return None

            # Extract location and timestamp
            lat_mean = float(ds.latitude.mean())
            lon_mean = float(ds.longitude.mean())

            if "time" in ds.coords:
                timestamp = str(ds.time.values[0])
            else:
                timestamp = datetime.utcnow().isoformat()

            granule_id = granule_meta.get("meta", {}).get("concept-id", "unknown")

            ds.close()

            return MethaneDetection(
                granule_id=granule_id,
                timestamp=timestamp,
                lat=lat_mean,
                lon=lon_mean,
                plume_size_pixels=plume_sum,
            )

        except Exception as e:
            logger.warning(f"Error processing granule: {e}")
            try:
                ds.close()
            except Exception:
                pass
            return None
