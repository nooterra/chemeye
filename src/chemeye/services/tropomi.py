"""
TROPOMI (Sentinel-5P) methane hotspot ingestion.

This pulls recent granules, filters for high-quality methane observations, and
extracts hotspot centroids for persistence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Tuple

import earthaccess
import numpy as np
import xarray as xr
from scipy.ndimage import label

logger = logging.getLogger(__name__)

TROPOMI_COLLECTION = "S5P_L2__CH4___"  # Sentinel-5P L2 Methane


def search_recent_tropomi_granules(hours: int = 24, count: int = 20):
    """Search TROPOMI granules in the last `hours` globally."""
    now = datetime.utcnow()
    start = (now - timedelta(hours=hours)).isoformat() + "Z"
    end = now.isoformat() + "Z"
    logger.info("Searching TROPOMI granules from %s to %s", start, end)
    granules = earthaccess.search_data(
        short_name=TROPOMI_COLLECTION,
        temporal=(start, end),
        count=count,
    )
    logger.info("Found %d TROPOMI granules", len(granules))
    return granules


def _extract_hotspots(ds: xr.Dataset, qa_min: float = 0.5, ch4_min: float = 1850.0) -> List[dict]:
    """
    Extract hotspot centroids from a TROPOMI dataset.

    Rules:
      - qa_value > qa_min
      - methane_mixing_ratio_bias_corrected > ch4_min (ppb)
      - Keep only connected components with >= 3 pixels.
    """
    required_vars = ["qa_value", "methane_mixing_ratio_bias_corrected", "latitude", "longitude"]
    for v in required_vars:
        if v not in ds:
            logger.warning("Missing variable %s in TROPOMI dataset", v)
            return []

    qa = ds["qa_value"].values
    ch4 = ds["methane_mixing_ratio_bias_corrected"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # Ensure shapes align
    if qa.shape != ch4.shape or qa.shape != lat.shape or qa.shape != lon.shape:
        logger.warning("Shape mismatch in TROPOMI variables")
        return []

    mask = (qa > qa_min) & (ch4 > ch4_min) & np.isfinite(ch4) & np.isfinite(lat) & np.isfinite(lon)
    if not mask.any():
        return []

    labeled, num_features = label(mask)
    hotspots: list[dict] = []

    for i in range(1, num_features + 1):
        component = labeled == i
        count = int(component.sum())
        if count < 3:
            continue

        # Centroid
        lat_vals = lat[component]
        lon_vals = lon[component]
        ch4_vals = ch4[component]

        centroid_lat = float(np.nanmean(lat_vals))
        centroid_lon = float(np.nanmean(lon_vals))
        mean_ppb = float(np.nanmean(ch4_vals))
        max_ppb = float(np.nanmax(ch4_vals))

        hotspots.append(
            {
                "lat": centroid_lat,
                "lon": centroid_lon,
                "pixel_count": count,
                "mean_ppb": mean_ppb,
                "max_ppb": max_ppb,
            }
        )

    logger.info("Extracted %d hotspots from TROPOMI granule", len(hotspots))
    return hotspots


def process_tropomi_granule(granule) -> List[dict]:
    """Open a TROPOMI granule and return hotspot dicts with metadata."""
    try:
        files = earthaccess.open([granule])
        if not files:
            logger.warning("No files returned for TROPOMI granule")
            return []
        ds = xr.open_dataset(files[0], engine="h5netcdf")
    except Exception as e:
        logger.warning(f"Failed to open TROPOMI granule: {e}")
        return []

    hotspots = _extract_hotspots(ds)

    # Timestamp from dataset or granule metadata
    timestamp = None
    if "time" in ds.coords:
        try:
            timestamp = np.datetime64(ds["time"].values).astype("datetime64[s]").astype(str)
        except Exception:
            timestamp = None
    if not timestamp:
        timestamp = datetime.utcnow().isoformat()

    granule_ur = granule["umm"]["GranuleUR"] if "umm" in granule else ""

    enriched = []
    for h in hotspots:
        h["timestamp"] = timestamp
        h["granule_ur"] = granule_ur
    return hotspots
