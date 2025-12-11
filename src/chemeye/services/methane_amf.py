"""
Methane detection engine using an Adaptive Matched Filter (AMF) on raw EMIT L2A
reflectance data.

Physics: Methane (CH4) has strong absorption features at ~2300nm. The AMF
suppresses background (terrain) and amplifies methane signatures using either
global or clustered background covariance models.
"""
import logging
from typing import Tuple, Dict, List
from dataclasses import dataclass

import xarray as xr
from scipy.linalg import pinv
from sklearn.cluster import KMeans

import numpy as np

from .preprocessing import destripe_cube, create_valid_pixel_mask
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
    wavelength_max: float = 2500.0,
    mode: str = "global",
    n_clusters: int = 10,
    iterative_background: bool = True,
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
    logger.info("🔬 Initializing Adaptive Matched Filter (global background)...")
    
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
    
    # Ensure 3D SWIR cube
    if swir.ndim != 3:
        raise ValueError(f"Expected 3D reflectance cube, got shape {swir.shape}")

    # Get wavelengths aligned with bands
    if 'wavelengths' in swir.coords:
        wavelengths = swir.coords['wavelengths'].values
        bands_dim = 'wavelengths'
    elif 'wavelength' in swir.coords:
        wavelengths = swir.coords['wavelength'].values
        bands_dim = 'wavelength'
    else:
        raise ValueError("Could not find wavelength coordinate")

    # Destripe SWIR cube
    swir_destriped = destripe_cube(swir)

    # Build strict valid pixel mask from SWIR bands
    valid_pixel_mask = create_valid_pixel_mask(swir, xr.DataArray(wavelengths, dims=(bands_dim,)))

    # Dimensions after possible transpose in destripe_cube
    h, w, bands = swir_destriped.values.shape
    
    logger.info(f"📊 Processing cube: {h}x{w} pixels × {bands} bands ({wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm)")
    
    # 2. Flatten and Clean Data
    X = swir_destriped.values.reshape(-1, bands)

    # Track valid pixels (not NaN and pass strict mask)
    flat_mask = valid_pixel_mask.reshape(-1)
    valid_mask = (~np.isnan(X).any(axis=1)) & flat_mask
    n_valid = valid_mask.sum()
    
    if n_valid < 100:
        logger.warning(f"Only {n_valid} valid pixels - insufficient for covariance estimation")
        return np.zeros((h, w)), {"error": "insufficient_data"}
    
    # Replace NaNs with 0 for matrix operations
    X_clean = np.nan_to_num(X, nan=0.0)
    
    def compute_background_stats(
        X_background: np.ndarray,
        labels: np.ndarray | None = None,
        n_clusters_local: int | None = None,
    ):
        """
        Compute background mean/covariance (global or per-cluster).

        Returns:
            mus: (K, bands) or (1, bands)
            covs: (K, bands, bands)
            cluster_labels: per-pixel cluster labels (for valid pixels), or None
        """
        if labels is None or n_clusters_local is None or n_clusters_local <= 1:
            mu_global = np.mean(X_background, axis=0)
            Sigma_global = np.cov(X_background, rowvar=False)
            return (
                mu_global[None, :],
                Sigma_global[None, :, :],
                None,
            )

        mus = []
        covs = []
        for k in range(n_clusters_local):
            k_mask = labels == k
            if not np.any(k_mask):
                # Fallback: approximate empty clusters with global stats later
                mus.append(np.mean(X_background, axis=0))
                covs.append(np.cov(X_background, rowvar=False))
                continue
            X_k = X_background[k_mask]
            mus.append(np.mean(X_k, axis=0))
            covs.append(np.cov(X_k, rowvar=False))

        return np.stack(mus, axis=0), np.stack(covs, axis=0), labels

    # Background modeling: global or cluster-based
    X_valid = X_clean[valid_mask]

    # KMeans clustering (on valid pixels) if requested
    cluster_labels = None
    effective_clusters = 1
    if mode == "cluster":
        max_clusters = max(1, min(n_clusters, int(n_valid // 1000) or 1))
        if max_clusters > 1:
            logger.info(f"Clustering background into {max_clusters} clusters for AMF")
            km = KMeans(n_clusters=max_clusters, n_init="auto", random_state=0)
            cluster_labels = km.fit_predict(X_valid)
            effective_clusters = max_clusters
        else:
            logger.info("Not enough valid pixels for clustering; falling back to global background")

    # Optionally perform iterative background cleaning (two-pass)
    def estimate_and_score(
        X_all: np.ndarray,
        valid_mask_all: np.ndarray,
        labels_valid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate background means/covariances and compute per-pixel Z-scores.

        Returns:
            z_map: (H, W) Z-score map
            alpha: (N,) raw filter response (flattened)
            mus: (K, bands)
            Sigmas: (K, bands, bands)
        """
        X_valid_local = X_all[valid_mask_all]

        mus, covs, labels_out = compute_background_stats(
            X_valid_local,
            labels_valid,
            effective_clusters if labels_valid is not None else None,
        )

        # Regularize and invert covariance per cluster
        target = get_methane_target_signature(wavelengths)
        alphas = np.zeros(X_all.shape[0], dtype=float)

        for k in range(mus.shape[0]):
            mu_k = mus[k]
            Sigma_k = covs[k]

            reg_strength = 1e-6 * np.trace(Sigma_k) / bands
            Sigma_reg = Sigma_k + reg_strength * np.eye(bands)
            try:
                Sigma_inv_k = pinv(Sigma_reg)
            except Exception as e:
                logger.error(f"Covariance inversion failed for cluster {k}: {e}")
                return (
                    np.zeros((h, w)),
                    np.zeros_like(alphas),
                    mus,
                    covs,
                )

            target_whitened = Sigma_inv_k @ target
            normalization = np.sqrt(target.T @ Sigma_inv_k @ target + 1e-10)

            if labels_out is None:
                # Global model: apply to all valid pixels
                idx_valid = np.where(valid_mask_all)[0]
                Xk = X_all[idx_valid]
                alphas[idx_valid] = (Xk - mu_k) @ target_whitened / normalization
            else:
                # Clustered model: apply to pixels in this cluster
                idx_valid = np.where(valid_mask_all)[0]
                cluster_mask = labels_out == k
                idx_cluster = idx_valid[cluster_mask]
                if idx_cluster.size == 0:
                    continue
                Xk = X_all[idx_cluster]
                alphas[idx_cluster] = (Xk - mu_k) @ target_whitened / normalization

        methane_map_local = alphas.reshape(h, w)

        # Z-score normalization on valid pixels only
        valid_values = methane_map_local[valid_pixel_mask]
        map_mean = np.nanmean(valid_values)
        map_std = np.nanstd(valid_values)

        z_map = (methane_map_local - map_mean) / (map_std + 1e-10)
        return z_map, alphas, mus, covs

    # First pass
    z_map_initial, alpha_initial, mus_initial, covs_initial = estimate_and_score(
        X_clean,
        valid_mask,
        cluster_labels,
    )

    # Iterative background cleaning: remove top 1% strongest pixels and recompute
    if iterative_background:
        flat_z_valid = z_map_initial.reshape(-1)[valid_mask]
        if flat_z_valid.size > 0:
            cutoff_index = max(int(0.99 * flat_z_valid.size), 0)
            z_threshold_background = np.partition(flat_z_valid, cutoff_index)[cutoff_index]
            logger.info(f"Iterative background: excluding pixels with z >= {z_threshold_background:.2f}")

            # Build background mask (valid and below cutoff)
            bg_mask_flat = valid_mask & (z_map_initial.reshape(-1) < z_threshold_background)

            z_map_final, alpha_final, mus_final, covs_final = estimate_and_score(
                X_clean,
                bg_mask_flat,
                cluster_labels,
            )
            z_score_map = z_map_final
            mus_used, covs_used = mus_final, covs_final
            max_z_initial = float(np.nanmax(z_map_initial))
        else:
            z_score_map = z_map_initial
            mus_used, covs_used = mus_initial, covs_initial
            max_z_initial = float(np.nanmax(z_map_initial))
    else:
        z_score_map = z_map_initial
        mus_used, covs_used = mus_initial, covs_initial
        max_z_initial = float(np.nanmax(z_map_initial))

    # Apply detection threshold
    detections = np.where(z_score_map > z_threshold, z_score_map, 0)

    max_z = float(np.nanmax(z_score_map))
    n_detections = int(np.sum(detections > 0))

    metadata = {
        "max_z_score": max_z,
        "initial_max_z_score": max_z_initial,
        "n_detection_pixels": n_detections,
        "threshold": z_threshold,
        "n_valid_pixels": int(n_valid),
        "wavelength_range_nm": (float(wavelengths[0]), float(wavelengths[-1])),
        "n_bands": bands,
        "mode": mode,
        "n_clusters": effective_clusters,
        "iterative_background": bool(iterative_background),
    }

    logger.info(
        "✅ AMF Complete. Mode=%s, clusters=%d, max Z: %.2fσ, pixels > threshold: %d",
        mode,
        effective_clusters,
        max_z,
        n_detections,
    )

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
