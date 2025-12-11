"""
Pre-processing utilities for EMIT L2A reflectance cubes.

Includes:
- Column-wise destriping to reduce push-broom detector artifacts
- Strict good-pixel masking for SWIR bands (water/cloud/saturation rejection)
"""

import logging
from typing import Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _resolve_dims(array: xr.DataArray) -> Tuple[str, str, str]:
    """
    Resolve (y, x, bands) dimension names for a 3D reflectance cube.

    Tries common EMIT L2A patterns and falls back to the existing order.
    """
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")

    dims = list(array.dims)

    # Prefer explicit ortho_y/ortho_x/bands if present
    if set(("ortho_y", "ortho_x", "bands")).issubset(dims):
        return "ortho_y", "ortho_x", "bands"

    # Common fallback: (y, x, band)
    if "bands" in dims or "band" in dims:
        bands_dim = "bands" if "bands" in dims else "band"
        other = [d for d in dims if d != bands_dim]
        if len(other) == 2:
            return other[0], other[1], bands_dim

    # Otherwise, assume current order is (y, x, bands)
    return dims[0], dims[1], dims[2]


def destripe_cube(reflectance: xr.DataArray) -> xr.DataArray:
    """
    Remove cross-track striping artifacts from a reflectance cube.

    For each band:
      - Compute global median over all valid pixels.
      - For each column (cross-track), compute its median.
      - Compute column_offset = column_median - global_median.
      - Subtract column_offset from every pixel in that column.

    Returns a new DataArray with the same shape and coordinates.
    """
    if reflectance.ndim != 3:
        raise ValueError(f"destripe_cube expects 3D data, got shape {reflectance.shape}")

    y_dim, x_dim, bands_dim = _resolve_dims(reflectance)
    data = reflectance.transpose(y_dim, x_dim, bands_dim).values  # (H, W, B)

    h, w, b = data.shape
    logger.info(f"Destriping cube: {h}x{w} pixels × {b} bands")

    destriped = data.copy()

    # Process each band independently to preserve spectral shape
    for bi in range(b):
        band_slice = destriped[:, :, bi]

        # Use only finite values for statistics
        finite_mask = np.isfinite(band_slice)
        if not finite_mask.any():
            continue

        global_median = np.median(band_slice[finite_mask])

        # Column medians along y-axis
        column_medians = np.median(
            np.where(finite_mask, band_slice, np.nan),
            axis=0,
        )

        # Where entire column is NaN, leave as-is
        valid_cols = np.isfinite(column_medians)
        offsets = np.zeros_like(column_medians)
        offsets[valid_cols] = column_medians[valid_cols] - global_median

        # Subtract per-column offsets
        destriped[:, :, bi] = band_slice - offsets[np.newaxis, :]

    return xr.DataArray(
        destriped,
        dims=(y_dim, x_dim, bands_dim),
        coords={
            y_dim: reflectance.coords.get(y_dim, np.arange(h)),
            x_dim: reflectance.coords.get(x_dim, np.arange(w)),
            bands_dim: reflectance.coords.get(bands_dim, np.arange(b)),
        },
        attrs=reflectance.attrs,
        name=reflectance.name,
    )


def create_valid_pixel_mask(
    reflectance: xr.DataArray,
    wavelengths: xr.DataArray,
    swir_min_nm: float = 2100.0,
) -> np.ndarray:
    """
    Build a strict boolean mask for "good" pixels using SWIR bands.

    Steps:
      - Select SWIR bands: wavelengths >= swir_min_nm.
      - For each pixel, compute the median reflectance over SWIR bands.
      - Mark pixel invalid if:
          swir_reflectance < 0.01  (water/shadow)
          OR swir_reflectance > 0.8 (cloud/saturation)

    Returns:
        2D boolean numpy array [y, x] where True = valid pixel.
    """
    if reflectance.ndim != 3:
        raise ValueError(f"create_valid_pixel_mask expects 3D data, got shape {reflectance.shape}")

    y_dim, x_dim, bands_dim = _resolve_dims(reflectance)

    # Align wavelengths with bands dimension
    if wavelengths.ndim != 1 or wavelengths.shape[0] != reflectance.sizes[bands_dim]:
        raise ValueError("Wavelengths must be 1D and match bands dimension length")

    swir_mask = wavelengths >= swir_min_nm
    if not bool(swir_mask.any()):
        logger.warning("No SWIR bands found above threshold; returning all-false mask")
        return np.zeros((reflectance.sizes[y_dim], reflectance.sizes[x_dim]), dtype=bool)

    swir = reflectance.sel({bands_dim: swir_mask})

    # Compute median across SWIR bands for each pixel
    swir_values = swir.transpose(y_dim, x_dim, bands_dim).values
    swir_median = np.nanmedian(swir_values, axis=2)

    # Build strict mask
    good = np.ones_like(swir_median, dtype=bool)

    # Too dark: water/shadow
    good &= swir_median >= 0.01

    # Too bright: clouds/saturation
    good &= swir_median <= 0.8

    # Exclude NaNs
    good &= np.isfinite(swir_median)

    logger.info(
        "Valid pixel mask: %d / %d pixels kept (%.2f%%)",
        int(good.sum()),
        int(good.size),
        100.0 * float(good.sum()) / float(good.size),
    )

    return good

