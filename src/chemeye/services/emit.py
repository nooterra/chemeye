"""
NASA EarthData authentication and EMIT data fetching.
"""

import logging
from typing import Optional

import earthaccess
import xarray as xr

from ..config import get_settings

logger = logging.getLogger(__name__)


# EMIT Product codes
METHANE_PRODUCT = "EMITL2BCH4PLM"  # Level 2B Methane Plume Complex
REFLECTANCE_PRODUCT = "EMITL2ARFL"  # Level 2A Surface Reflectance


class EMITService:
    """Service for accessing NASA EMIT satellite data."""

    def __init__(self):
        self._authenticated = False

    def authenticate(self) -> bool:
        """
        Authenticate with NASA EarthData.

        Uses credentials from environment or falls back to .netrc.
        """
        if self._authenticated:
            return True

        settings = get_settings()

        try:
            if settings.nasa_earthdata_username and settings.nasa_earthdata_password:
                # earthaccess expects EARTHDATA_USERNAME/PASSWORD env vars
                import os
                os.environ["EARTHDATA_USERNAME"] = settings.nasa_earthdata_username
                os.environ["EARTHDATA_PASSWORD"] = settings.nasa_earthdata_password
                
                logger.info("Authenticating with NASA EarthData using env credentials...")
                earthaccess.login(
                    strategy="environment",
                    persist=True,
                )
            else:
                # Fall back to netrc
                logger.info("Authenticating with NASA EarthData using .netrc...")
                earthaccess.login(strategy="netrc", persist=True)

            self._authenticated = True
            logger.info("NASA EarthData authentication successful")
            return True

        except Exception as e:
            logger.error(f"NASA EarthData authentication failed: {e}")
            return False

    def search_methane_granules(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        count: int = 10,
    ) -> list:
        """
        Search for EMIT methane plume granules.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            count: Maximum number of granules to return

        Returns:
            List of granule metadata
        """
        if not self._authenticated:
            self.authenticate()

        logger.info(f"Searching EMIT methane granules: bbox={bbox}, dates={start_date} to {end_date}")

        results = earthaccess.search_data(
            short_name=METHANE_PRODUCT,
            bounding_box=bbox,
            temporal=(start_date, end_date),
            count=count,
        )

        logger.info(f"Found {len(results)} methane granules")
        return results

    def search_reflectance_granules(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        count: int = 1,
    ) -> list:
        """
        Search for EMIT reflectance granules.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            count: Maximum number of granules to return

        Returns:
            List of granule metadata
        """
        if not self._authenticated:
            self.authenticate()

        logger.info(f"Searching EMIT reflectance granules: bbox={bbox}, dates={start_date} to {end_date}")

        results = earthaccess.search_data(
            short_name=REFLECTANCE_PRODUCT,
            bounding_box=bbox,
            temporal=(start_date, end_date),
            count=count,
        )

        logger.info(f"Found {len(results)} reflectance granules")
        return results

    def open_granules(self, granules: list) -> list:
        """
        Open granules for streaming access.
        
        Filters to only return NetCDF (.nc) files, skipping JSON/GeoTIFF.

        Args:
            granules: List of granule metadata from search

        Returns:
            List of file-like objects for streaming (NetCDF only)
        """
        if not granules:
            return []

        logger.info(f"Opening {len(granules)} granules for streaming...")
        all_files = earthaccess.open(granules)
        
        # Filter for NetCDF files only
        nc_files = []
        for f in all_files:
            try:
                # Get the filename from the file object
                name = getattr(f, 'path', getattr(f, 'name', str(f)))
                if name.endswith('.nc'):
                    nc_files.append(f)
                    logger.debug(f"Including NetCDF file: {name}")
                else:
                    logger.debug(f"Skipping non-NetCDF file: {name}")
            except Exception as e:
                logger.debug(f"Could not determine file type, including: {e}")
                nc_files.append(f)  # Include if we can't determine
        
        logger.info(f"Filtered to {len(nc_files)} NetCDF files from {len(all_files)} total files")
        return nc_files

    def load_dataset(self, file_obj, engine: str = "h5netcdf") -> Optional[xr.Dataset]:
        """
        Load a granule as an xarray Dataset.

        Args:
            file_obj: File object from open_granules
            engine: xarray engine to use

        Returns:
            xarray Dataset or None on error
        """
        try:
            return xr.open_dataset(file_obj, engine=engine, chunks="auto")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None


# Singleton instance
_emit_service: Optional[EMITService] = None


def get_emit_service() -> EMITService:
    """Get singleton EMIT service instance."""
    global _emit_service
    if _emit_service is None:
        _emit_service = EMITService()
    return _emit_service
