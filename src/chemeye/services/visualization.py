"""
Visualization utilities for methane detection.

Provides:
- RGB composite generation from EMIT L2A reflectance cubes
- Overlay rendering of AMF Z-score maps on RGB for human inspection
- Helper to write a Z-score GeoTIFF suitable for web map tooling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _resolve_wavelength_coords(ds: xr.Dataset) -> Tuple[xr.DataArray, str]:
    """
    Resolve wavelength coordinate and its dimension name.

    Returns:
        (wavelengths, dim_name)
    """
    if "wavelengths" in ds.coords:
        return ds.coords["wavelengths"], "wavelengths"
    if "wavelength" in ds.coords:
        return ds.coords["wavelength"], "wavelength"
    if "wavelengths" in ds.data_vars:
        ds = ds.set_coords("wavelengths")
        return ds.coords["wavelengths"], "wavelengths"
    if "wavelength" in ds.data_vars:
        ds = ds.set_coords("wavelength")
        return ds.coords["wavelength"], "wavelength"
    raise ValueError("Could not find wavelength coordinate in dataset")


def get_rgb_array(reflectance_ds: xr.Dataset) -> np.ndarray:
    """
    Build an 8-bit RGB composite from an EMIT L2A reflectance dataset.

    Band selection (approximate natural color):
      - Red   ~ 650 nm
      - Green ~ 560 nm
      - Blue  ~ 470 nm

    A 2–98 percentile stretch is applied across all channels and the result
    is scaled to [0, 255] uint8.
    """
    if "reflectance" not in reflectance_ds:
        raise ValueError("Dataset must contain 'reflectance' variable")

    refl = reflectance_ds["reflectance"]

    wavelengths, wave_dim = _resolve_wavelength_coords(reflectance_ds)

    def _sel_band(target_nm: float) -> xr.DataArray:
        try:
            return refl.sel({wave_dim: target_nm}, method="nearest")
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Failed to select band near {target_nm} nm: {e}") from e

    red_da = _sel_band(650.0)
    green_da = _sel_band(560.0)
    blue_da = _sel_band(470.0)

    # Ensure consistent dimension order (y, x)
    if red_da.ndim != 2 or green_da.ndim != 2 or blue_da.ndim != 2:
        raise ValueError("Expected 2D reflectance bands for RGB composite")

    red = red_da.values.astype(np.float32)
    green = green_da.values.astype(np.float32)
    blue = blue_da.values.astype(np.float32)

    rgb = np.stack([red, green, blue], axis=-1)  # (H, W, 3)

    # Percentile stretch across all channels
    flat = rgb.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    if not valid.any():
        raise ValueError("No finite reflectance values available for RGB composite")

    vmin = np.percentile(flat[valid], 2)
    vmax = np.percentile(flat[valid], 98)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    scaled = np.clip((rgb - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb_uint8 = (scaled * 255.0).astype(np.uint8)

    logger.info(
        "Generated RGB composite with percentile stretch [%.4f, %.4f]",
        vmin,
        vmax,
    )
    return rgb_uint8


def generate_overlay_image(
    rgb_array: np.ndarray,
    z_score_map: np.ndarray,
    output_path: str | Path,
    threshold: float = 3.0,
) -> None:
    """
    Render an RGB+heatmap overlay as a PNG.

    Args:
        rgb_array: (H, W, 3) uint8 RGB image.
        z_score_map: (H, W) float array of AMF Z-scores.
        output_path: Destination PNG path.
        threshold: Z-score threshold at which to begin rendering the plume.
    """
    from matplotlib import cm
    from matplotlib import pyplot as plt

    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected RGB array shape (H, W, 3), got {rgb_array.shape}")
    if z_score_map.shape != rgb_array.shape[:2]:
        raise ValueError("z_score_map shape must match RGB spatial dimensions")

    h, w, _ = rgb_array.shape

    rgb_f = rgb_array.astype(np.float32) / 255.0
    z = np.asarray(z_score_map, dtype=np.float32)

    if not np.isfinite(z).any():
        logger.warning("Z-score map contains no finite values; saving RGB only")
        plt.imsave(str(output_path), rgb_f)
        return

    max_z = float(np.nanmax(z))
    if max_z <= threshold:
        logger.info(
            "Max Z-score %.2f is below threshold %.2f; overlay will be very faint",
            max_z,
            threshold,
        )

    # Normalize Z-scores into [0, 1] for colormap
    denom = max(max_z - threshold, 1e-3)
    z_norm = np.clip((z - threshold) / denom, 0.0, 1.0)

    cmap = cm.get_cmap("plasma")
    heat_rgb = cmap(z_norm)[..., :3]  # (H, W, 3)

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[z >= threshold] = 0.6
    alpha = alpha[..., None]  # (H, W, 1)

    overlay = alpha * heat_rgb + (1.0 - alpha) * rgb_f
    overlay = np.clip(overlay, 0.0, 1.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(str(output_path), overlay)
    logger.info("Saved overlay image to %s", output_path)


def write_z_score_cog(
    z_score_map: np.ndarray,
    output_path: str | Path,
    lon_array: Optional[np.ndarray] = None,
    lat_array: Optional[np.ndarray] = None,
) -> None:
    """
    Write a Z-score map to a GeoTIFF with tiling and compression.

    If lat/lon arrays are provided (2D, same shape as z_score_map), they are
    used to derive an approximate affine transform in EPSG:4326 so that the
    raster aligns with maps in tools like QGIS or deck.gl. Otherwise, a
    placeholder transform is used.
    """
    import rasterio
    from rasterio.transform import from_origin

    z = np.asarray(z_score_map, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D z_score_map, got shape {z.shape}")

    height, width = z.shape

    # Derive transform from lat/lon grids if available
    if lon_array is not None and lat_array is not None:
        lon = np.asarray(lon_array, dtype=float)
        lat = np.asarray(lat_array, dtype=float)
        if lon.shape != z.shape or lat.shape != z.shape:
            raise ValueError("lat_array and lon_array must match z_score_map shape")

        min_lon = float(np.nanmin(lon))
        max_lon = float(np.nanmax(lon))
        min_lat = float(np.nanmin(lat))
        max_lat = float(np.nanmax(lat))

        # Approximate pixel sizes in degrees
        xres = (max_lon - min_lon) / max(width, 1)
        yres = (max_lat - min_lat) / max(height, 1)
        yres = abs(yres)  # from_origin expects positive pixel height

        transform = from_origin(min_lon, max_lat, xres, yres)
        logger.info(
            "Using georeferenced transform from lat/lon grids: "
            "west=%.6f east=%.6f south=%.6f north=%.6f, xres=%.6g, yres=%.6g",
            min_lon,
            max_lon,
            min_lat,
            max_lat,
            xres,
            yres,
        )
    else:
        # Fallback placeholder transform
        transform = from_origin(0.0, 0.0, 1.0, 1.0)
        logger.warning("No lat/lon grids provided; writing Z-score COG with placeholder transform")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "tiled": True,
        "compress": "deflate",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(z, 1)

    logger.info("Saved Z-score GeoTIFF to %s", output_path)
