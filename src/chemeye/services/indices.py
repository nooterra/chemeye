"""
Pure math functions for spectral index calculations.
No I/O, no EMIT dependencies — just arrays in, values out.
"""

import numpy as np
from numpy.typing import ArrayLike


def ndvi(red: ArrayLike, nir: ArrayLike) -> ArrayLike:
    """
    Calculate Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        red: Red band reflectance (~650nm)
        nir: Near-infrared band reflectance (~850nm)

    Returns:
        NDVI values in range [-1, 1]
    """
    red = np.asarray(red)
    nir = np.asarray(nir)

    # Avoid division by zero
    denominator = nir + red
    mask = denominator != 0

    result = np.zeros_like(denominator, dtype=np.float32)
    result[mask] = (nir[mask] - red[mask]) / denominator[mask]

    return result


def ndni(nir_1510: ArrayLike, nir_1680: ArrayLike) -> ArrayLike:
    """
    Calculate Normalized Difference Nitrogen Index.

    NDNI = (log(1/R1510) - log(1/R1680)) / (log(1/R1510) + log(1/R1680))

    Args:
        nir_1510: Reflectance at ~1510nm
        nir_1680: Reflectance at ~1680nm

    Returns:
        NDNI values (higher = more nitrogen stress)
    """
    nir_1510 = np.asarray(nir_1510, dtype=np.float32)
    nir_1680 = np.asarray(nir_1680, dtype=np.float32)

    # Avoid log(0) and division by zero
    eps = 1e-6
    nir_1510 = np.clip(nir_1510, eps, None)
    nir_1680 = np.clip(nir_1680, eps, None)

    log_1510 = np.log(1.0 / nir_1510)
    log_1680 = np.log(1.0 / nir_1680)

    denominator = log_1510 + log_1680
    mask = np.abs(denominator) > eps

    result = np.zeros_like(denominator)
    result[mask] = (log_1510[mask] - log_1680[mask]) / denominator[mask]

    return result


def clay_absorption_depth(
    r2120: ArrayLike,
    r2200: ArrayLike,
    r2280: ArrayLike,
) -> ArrayLike:
    """
    Calculate clay mineral absorption depth at ~2200nm.

    Uses continuum removal approach:
    depth = 1 - (R2200 / continuum)
    where continuum is interpolated between R2120 and R2280

    Args:
        r2120: Reflectance at ~2120nm (shoulder)
        r2200: Reflectance at ~2200nm (absorption)
        r2280: Reflectance at ~2280nm (shoulder)

    Returns:
        Absorption depth (higher = stronger clay signature)
    """
    r2120 = np.asarray(r2120, dtype=np.float32)
    r2200 = np.asarray(r2200, dtype=np.float32)
    r2280 = np.asarray(r2280, dtype=np.float32)

    # Interpolate continuum at 2200nm
    # 2200 is at (2200-2120)/(2280-2120) = 0.5 between shoulders
    continuum = (r2120 + r2280) / 2.0

    # Avoid division by zero
    eps = 1e-6
    continuum = np.clip(continuum, eps, None)

    depth = 1.0 - (r2200 / continuum)

    # Clip to valid range
    return np.clip(depth, 0.0, 1.0)


def iron_oxide_ratio(red: ArrayLike, blue: ArrayLike) -> ArrayLike:
    """
    Calculate iron oxide ratio (ferric iron indicator).

    Ratio = Red / Blue

    Higher values indicate iron oxide presence.

    Args:
        red: Red band reflectance (~650nm)
        blue: Blue band reflectance (~450nm)

    Returns:
        Iron oxide ratio
    """
    red = np.asarray(red, dtype=np.float32)
    blue = np.asarray(blue, dtype=np.float32)

    # Avoid division by zero
    eps = 1e-6
    blue = np.clip(blue, eps, None)

    return red / blue


def methane_absorption_depth(
    r2200: ArrayLike,
    r2300: ArrayLike,
    r2400: ArrayLike,
) -> ArrayLike:
    """
    Calculate methane absorption depth at ~2300nm.

    Uses continuum removal approach similar to clay.

    Args:
        r2200: Reflectance at ~2200nm (shoulder)
        r2300: Reflectance at ~2300nm (absorption)
        r2400: Reflectance at ~2400nm (shoulder)

    Returns:
        Absorption depth (higher = stronger methane signature)
    """
    r2200 = np.asarray(r2200, dtype=np.float32)
    r2300 = np.asarray(r2300, dtype=np.float32)
    r2400 = np.asarray(r2400, dtype=np.float32)

    # Interpolate continuum at 2300nm
    continuum = (r2200 + r2400) / 2.0

    # Avoid division by zero
    eps = 1e-6
    continuum = np.clip(continuum, eps, None)

    depth = 1.0 - (r2300 / continuum)

    return np.clip(depth, 0.0, 1.0)


class SpectralIndices:
    """Container for all spectral index functions."""

    ndvi = staticmethod(ndvi)
    ndni = staticmethod(ndni)
    clay_absorption_depth = staticmethod(clay_absorption_depth)
    iron_oxide_ratio = staticmethod(iron_oxide_ratio)
    methane_absorption_depth = staticmethod(methane_absorption_depth)
