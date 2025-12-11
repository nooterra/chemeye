"""
Sentinel-5P (TROPOMI) methane hotspot ingestion.

This module searches daily granules, filters high-quality methane pixels, and
clusters them into hotspots suitable for populating the global map (yellow
layer). It intentionally keeps the interface minimal so workers can reuse it.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

import earthaccess
import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

# ESA/NASA short name for Sentinel-5P methane L2
COLLECTION_SHORTNAME = "S5P_L2__CH4___"

# Thresholds
BACKGROUND_THRESHOLD = 1850.0  # ppb
QUALITY_THRESHOLD = 0.5  # qa_value


def _load_granule_dataset(granule) -> Optional[xr.Dataset]:
    """Open a single TROPOMI granule via earthaccess."""
    try:
        files = earthaccess.open([granule])
        if not files:
            logger.warning("No files returned for TROPOMI granule")
            return None
        # h5netcdf is the usual engine for S5P L2
        return xr.open_dataset(files[0], engine="h5netcdf")
    except Exception as e:
        logger.warning("Failed to open TROPOMI granule: %s", e)
        return None


def _cluster_hotspots(lat: np.ndarray, lon: np.ndarray, ch4: np.ndarray) -> List[dict]:
    """
    Cluster high methane pixels into hotspots using DBSCAN.
    eps is in degrees (~5km). min_samples=3 filters speckle noise.
    """
    coords = np.column_stack((lon, lat))  # lon, lat order for eps degrees
    if coords.shape[0] == 0:
        return []

    model = DBSCAN(eps=0.05, min_samples=3, metric="euclidean")
    labels = model.fit_predict(coords)
    hotspots: list[dict] = []

    for label in np.unique(labels):
        if label == -1:
            continue  # noise
        mask = labels == label
        if not np.any(mask):
            continue

        lons = lon[mask]
        lats = lat[mask]
        vals = ch4[mask]

        centroid_lon = float(np.nanmean(lons))
        centroid_lat = float(np.nanmean(lats))
        mean_ppb = float(np.nanmean(vals))
        max_ppb = float(np.nanmax(vals))
        count = int(mask.sum())

        # Map ppb to a rough z-score-ish scale for consistent UI intensity
        z_score = max(0.0, (max_ppb - 1800.0) / 10.0)

        hotspots.append(
            {
                "lat": centroid_lat,
                "lon": centroid_lon,
                "pixel_count": count,
                "mean_ppb": mean_ppb,
                "max_ppb": max_ppb,
                "z_score": z_score,
            }
        )

    return hotspots


def _extract_hotspots(ds: xr.Dataset) -> List[dict]:
    """Filter QA, apply background threshold, and cluster."""
    required = ["qa_value", "methane_mixing_ratio_bias_corrected", "latitude", "longitude"]
    if not all(v in ds for v in required):
        missing = [v for v in required if v not in ds]
        logger.warning("TROPOMI dataset missing variables: %s", missing)
        return []

    qa = ds["qa_value"].values
    ch4 = ds["methane_mixing_ratio_bias_corrected"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    mask = (
        (qa > QUALITY_THRESHOLD)
        & (ch4 > BACKGROUND_THRESHOLD)
        & np.isfinite(ch4)
        & np.isfinite(lat)
        & np.isfinite(lon)
    )
    if not mask.any():
        return []

    return _cluster_hotspots(lat[mask], lon[mask], ch4[mask])


def search_daily_hotspots(day: date) -> List[dict]:
    """
    Find TROPOMI methane hotspots for a given UTC date.

    Returns a list of dicts with lat/lon/z_score/ppb metadata plus granule/timestamp.
    """
    start = datetime.combine(day, datetime.min.time()).isoformat() + "Z"
    end = (datetime.combine(day, datetime.min.time()) + timedelta(days=1)).isoformat() + "Z"
    granules = earthaccess.search_data(short_name=COLLECTION_SHORTNAME, temporal=(start, end))

    logger.info("Found %d TROPOMI granules for %s", len(granules), day.isoformat())

    hotspots: list[dict] = []

    for g in granules:
        ds = _load_granule_dataset(g)
        if ds is None:
            continue

        granule_ur = g.get("umm", {}).get("GranuleUR", "")

        ts = None
        if "time" in ds.coords:
            try:
                ts = np.datetime64(ds["time"].values).astype("datetime64[s]").astype(str)
            except Exception:
                ts = None
        if not ts:
            ts = datetime.utcnow().isoformat()

        for h in _extract_hotspots(ds):
            h["granule_ur"] = granule_ur
            h["timestamp"] = ts
            hotspots.append(h)

        ds.close()

    logger.info("Extracted %d hotspots for %s", len(hotspots), day.isoformat())
    return hotspots
