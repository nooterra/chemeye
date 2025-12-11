"""
SOTA Methane Detection Engine
Uses Adaptive Matched Filter (AMF) on raw EMIT L2A reflectance data.

Physics: Methane (CH4) has strong absorption features at ~2300nm.
The AMF suppresses background (terrain) and amplifies methane signatures.
"""
import numpy as np
import xarray as xr
from scipy.linalg import pinv
import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PlumeDetection:
    """Represents a detected methane plume"""
    lat: float
    lon: float
    z_score: float  # Signal strength in standard deviations
    pixel_count: int
    centroid_row: int
    centroid_col: int


def get_methane_target_signature(wavelengths: np.ndarray) -> np.ndarray:
    """
    Creates a synthetic absorption vector for Methane (CH4).
    
    Physics: CH4 has strong absorption features around 2.3 microns (2300nm).
    This creates a "unit absorption" profile for the matched filter.
    
    Args:
        wavelengths: Array of wavelengths in nanometers
        
    Returns:
        Target signature vector (same shape as wavelengths)
    """
    target = np.zeros_like(wavelengths, dtype=float)
    
    # Primary methane absorption band: 2150-2450 nm
    # In production, this would come from HITRAN spectroscopy database
    # For MVP, we use a simple box model
    mask = (wavelengths >= 2150) & (wavelengths <= 2450)
    target[mask] = 1.0
    
    # Normalize to unit vector
    norm = np.linalg.norm(target)
    if norm > 0:
        target = target / norm
    
    return target


def run_matched_filter(
    reflectance_ds: xr.Dataset,
    z_threshold: float = 3.0,
    wavelength_min: float = 2100.0,
    wavelength_max: float = 2500.0
) -> Tuple[np.ndarray, Dict]:
    """
    Runs the Adaptive Matched Filter (AMF) on raw L2A reflectance data.
    
    The AMF is a statistical algorithm that:
    1. Learns the "background" (Earth terrain) from the image itself
    2. Whitens the data to suppress common features
    3. Projects onto the methane absorption axis
    4. Returns Z-scores (signal strength above noise)
    
    Args:
        reflectance_ds: xarray Dataset with 'reflectance' variable
        z_threshold: Detection threshold in standard deviations
        wavelength_min: Minimum wavelength to use (nm)
        wavelength_max: Maximum wavelength to use (nm)
        
    Returns:
        Tuple of (detection_map, metadata_dict)
    """
    logger.info("🔬 SOTA: Initializing Adaptive Matched Filter...")
    
    # Validate input
    if 'reflectance' not in reflectance_ds:
        raise ValueError("Dataset must contain 'reflectance' variable")
    
    # Robust wavelength coordinate handling
    # EMIT L2A files sometimes hide wavelengths in data_vars, not coords
    if 'wavelengths' in reflectance_ds.data_vars:
        reflectance_ds = reflectance_ds.set_coords('wavelengths')
    elif 'wavelength' in reflectance_ds.data_vars:
        reflectance_ds = reflectance_ds.set_coords('wavelength')
    
    # Check if wavelengths are now available
    valid_coords = [c for c in reflectance_ds.coords if 'wave' in c]
    if not valid_coords:
        # If still missing, we might have to use local index (assuming standard EMIT bands)
        logger.warning("No wavelength coordinate found! Assuming standard EMIT 285 bands.")
        # Create dummy wavelengths if needed, or fail gracefully
        # For now, let's try to proceed by looking at dimensions
        if 'bands' in reflectance_ds.dims:
            # Create synthetic wavelengths 380-2500nm
            n_bands = reflectance_ds.dims['bands']
            synth_waves = np.linspace(380, 2500, n_bands)
            reflectance_ds['wavelengths'] = (('bands',), synth_waves)
            reflectance_ds = reflectance_ds.set_coords('wavelengths')

    # 1. Select Shortwave Infrared (SWIR) Bands
    # Methane is invisible in visible light - only use 2100-2500nm
    try:
        swir = reflectance_ds['reflectance'].sel(
            wavelengths=slice(wavelength_min, wavelength_max)
        )
    except KeyError:
        # Try alternative dimension name
        swir = reflectance_ds['reflectance'].sel(
            wavelength=slice(wavelength_min, wavelength_max)
        )
    
    # Get dimensions
    if len(swir.shape) == 3:
        h, w, bands = swir.shape
    else:
        raise ValueError(f"Expected 3D reflectance cube, got shape {swir.shape}")
    
    # Get wavelengths
    if 'wavelengths' in swir.coords:
        wavelengths = swir.wavelengths.values
    elif 'wavelength' in swir.coords:
        wavelengths = swir.wavelength.values
    else:
        raise ValueError("Could not find wavelength coordinate")
    
    logger.info(f"📊 Processing cube: {h}x{w} pixels × {bands} bands ({wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm)")
    
    # 2. Flatten and Clean Data
    X = swir.values.reshape(-1, bands)
    
    # Track valid pixels (not NaN/cloud)
    valid_mask = ~np.isnan(X).any(axis=1)
    n_valid = valid_mask.sum()
    
    if n_valid < 100:
        logger.warning(f"Only {n_valid} valid pixels - insufficient for covariance estimation")
        return np.zeros((h, w)), {"error": "insufficient_data"}
    
    # Replace NaNs with 0 for matrix operations
    X_clean = np.nan_to_num(X, nan=0.0)
    
    # 3. Compute Background Statistics (The "Adaptive" Part)
    # Use only valid pixels for statistics
    X_valid = X_clean[valid_mask]
    mu = np.mean(X_valid, axis=0)
    
    logger.info(f"Computing covariance from {n_valid:,} valid pixels...")
    
    # For very large images, subsample for covariance
    if n_valid > 50000:
        sample_idx = np.random.choice(n_valid, size=50000, replace=False)
        X_sample = X_valid[sample_idx]
        logger.info("Using 50k subsample for covariance estimation")
    else:
        X_sample = X_valid
    
    # Covariance matrix (bands × bands)
    Sigma = np.cov(X_sample, rowvar=False)
    
    # 4. Inverse Covariance (Whitening)
    # Add regularization for numerical stability
    reg_strength = 1e-6 * np.trace(Sigma) / bands
    Sigma_reg = Sigma + reg_strength * np.eye(bands)
    
    try:
        Sigma_inv = pinv(Sigma_reg)
    except Exception as e:
        logger.error(f"Covariance inversion failed: {e}")
        return np.zeros((h, w)), {"error": "inversion_failed"}
    
    # 5. Define Target (Methane absorption signature)
    t = get_methane_target_signature(wavelengths)
    
    # 6. The Matched Filter Equation
    # MF(x) = ((x - mu)^T · Sigma_inv · t) / sqrt(t^T · Sigma_inv · t)
    
    # Pre-compute target projection
    target_whitened = Sigma_inv @ t
    normalization = np.sqrt(t.T @ Sigma_inv @ t + 1e-10)
    
    # Apply to all pixels (vectorized)
    alpha = (X_clean - mu) @ target_whitened / normalization
    
    # Reshape to map
    methane_map = alpha.reshape(h, w)
    
    # 7. Z-Score Normalization
    # Convert to "standard deviations above background"
    valid_map = methane_map[~np.isnan(methane_map)]
    map_mean = np.nanmean(valid_map)
    map_std = np.nanstd(valid_map)
    
    z_score_map = (methane_map - map_mean) / (map_std + 1e-10)
    
    # 8. Apply threshold
    detections = np.where(z_score_map > z_threshold, z_score_map, 0)
    
    # Compute metadata
    max_z = np.max(z_score_map)
    n_detections = np.sum(detections > 0)
    
    metadata = {
        "max_z_score": float(max_z),
        "n_detection_pixels": int(n_detections),
        "threshold": z_threshold,
        "n_valid_pixels": int(n_valid),
        "wavelength_range_nm": (float(wavelengths[0]), float(wavelengths[-1])),
        "n_bands": bands
    }
    
    logger.info(f"✅ AMF Complete. Max Z-Score: {max_z:.2f}σ, Pixels > threshold: {n_detections:,}")
    
    return detections, metadata


def extract_plume_locations(
    detection_map: np.ndarray,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    min_pixels: int = 5
) -> List[PlumeDetection]:
    """
    Extracts discrete plume locations from detection map.
    
    Uses connected component analysis to group nearby detections.
    
    Args:
        detection_map: 2D array of Z-scores
        lat_array: 2D array of latitudes for each pixel
        lon_array: 2D array of longitudes for each pixel
        min_pixels: Minimum pixels for a valid plume
        
    Returns:
        List of PlumeDetection objects
    """
    from scipy.ndimage import label
    
    # Binary mask of detections
    binary_mask = detection_map > 0
    
    # Find connected components
    labeled_array, n_features = label(binary_mask)
    
    plumes = []
    for feature_id in range(1, n_features + 1):
        feature_mask = labeled_array == feature_id
        pixel_count = feature_mask.sum()
        
        if pixel_count < min_pixels:
            continue
        
        # Get centroid
        rows, cols = np.where(feature_mask)
        centroid_row = int(np.mean(rows))
        centroid_col = int(np.mean(cols))
        
        # Get max Z-score in this plume
        z_score = detection_map[feature_mask].max()
        
        # Get geographic coordinates
        lat = float(lat_array[centroid_row, centroid_col])
        lon = float(lon_array[centroid_row, centroid_col])
        
        plumes.append(PlumeDetection(
            lat=lat,
            lon=lon,
            z_score=z_score,
            pixel_count=pixel_count,
            centroid_row=centroid_row,
            centroid_col=centroid_col
        ))
    
    # Sort by Z-score (strongest first)
    plumes.sort(key=lambda p: p.z_score, reverse=True)
    
    logger.info(f"Extracted {len(plumes)} plume features")
    
    return plumes
