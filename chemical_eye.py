import argparse
import json
import sys
from typing import Optional

import earthaccess
import matplotlib.pyplot as plt
import requests
import xarray as xr


# --- CONFIGURATION ---
METHANE_PRODUCT = "EMITL2BCH4PLM"  # Level 2B Methane Plume Complex
REFLECTANCE_PRODUCT = "EMITL2ARFL"  # Level 2A Surface Reflectance


def setup_auth():
    """Authenticates with NASA EarthData. Uses .netrc if available; falls back to interactive."""
    print("🔐 Authenticating with NASA EarthData...")
    try:
        return earthaccess.login(strategy="netrc", persist=True)
    except Exception:
        print("ℹ️ netrc authentication failed; falling back to interactive prompt.")
        return earthaccess.login(strategy="interactive", persist=True)


def run_methane_hunter(bbox, date_range, *, plot: bool = True, webhook: Optional[str] = None):
    """
    MODULE 1: Search for CONFIRMED methane plumes (L2B Product).
    """
    print(f"🔎 Hunting for Methane in {bbox} between {date_range}...")

    results = earthaccess.search_data(
        short_name=METHANE_PRODUCT,
        bounding_box=bbox,
        temporal=date_range,
        count=10,
    )

    if not results:
        return {"status": "EMPTY", "message": "No EMIT passes found over this location/time."}

    print(f"📡 Found {len(results)} granules. Scanning for plumes...")

    # Stream the data (Lazy load)
    files = earthaccess.open(results)

    detections: list[dict] = []

    for i, file_obj in enumerate(files):
        try:
            # Open dataset using xarray with h5netcdf engine
            ds = xr.open_dataset(file_obj, engine="h5netcdf", chunks="auto")

            # The mask variable is 'ch4_plume_complex'
            # 1 = Plume, 0 = No Plume (Simplified)
            if "ch4_plume_complex" in ds.variables:
                plume_sum = ds["ch4_plume_complex"].sum().compute()

                if plume_sum > 0:
                    # Metadata extraction
                    lat_mean = float(ds.latitude.mean())
                    lon_mean = float(ds.longitude.mean())
                    timestamp = str(ds.time.values[0]) if "time" in ds.coords else "Unknown"

                    print(
                        f"🚨 DETECTED: Plume confirmed in Granule {i} | Size: {int(plume_sum)} pixels"
                    )

                    detection = {
                        "status": "DETECTED",
                        "granule_id": results[i]["meta"]["concept-id"],
                        "timestamp": timestamp,
                        "location": {"lat": lat_mean, "lon": lon_mean},
                        "plume_size_pixels": int(plume_sum),
                    }
                    detections.append(detection)

                    # Optional: Plot the finding
                    if plot:
                        plt.figure(figsize=(10, 6))
                        ds["ch4_plume_complex"].plot(cmap="YlOrRd")
                        plt.title(
                            f"CONFIRMED METHANE PLUME\n{timestamp} | Lat: {lat_mean:.3f}, Lon: {lon_mean:.3f}"
                        )
                        plt.show()

            ds.close()

        except Exception as e:  # pragma: no cover - best effort logging
            print(f"⚠️ Error reading granule {i}: {e}")

    result = (
        {"status": "CLEAR", "scanned_count": len(results)}
        if not detections
        else {"status": "DETECTED", "count": len(detections), "results": detections}
    )

    if not detections:
        print("✅ Clean scan. No plumes detected in searched granules.")
    else:
        print(f"📨 Prepared detection payload for {len(detections)} detections.")

    if webhook:
        try:
            resp = requests.post(webhook, json=result, timeout=10)
            resp.raise_for_status()
            print(f"📤 Posted detection payload to webhook ({resp.status_code}).")
        except Exception as exc:  # pragma: no cover - network path
            print(f"⚠️ Failed to post to webhook: {exc}")

    return result


def run_spectral_analysis(bbox, date_range, *, plot: bool = True):
    """
    MODULE 2: Raw Chemistry Analysis (NDVI + Spectral Fingerprint).
    """
    print(f"🔬 Analyzing Spectral Chemistry in {bbox}...")

    results = earthaccess.search_data(
        short_name=REFLECTANCE_PRODUCT,
        bounding_box=bbox,
        temporal=date_range,
        count=1,
    )

    if not results:
        print("❌ No L2A Reflectance data found.")
        return

    print(f"📉 Streaming Granule: {results[0]['meta']['concept-id']}...")
    files = earthaccess.open(results)
    ds = xr.open_dataset(files[0], engine="h5netcdf")

    # Pick a center pixel for the demo
    center_x = ds.dims["ortho_x"] // 2
    center_y = ds.dims["ortho_y"] // 2

    # Extract Spectrum
    spectrum = ds["reflectance"].isel(ortho_x=center_x, ortho_y=center_y)
    wavelengths = ds["wavelengths"].values

    # Calculate NDVI (Vegetation Index)
    # NIR ~ 850nm, Red ~ 650nm
    try:
        nir = ds["reflectance"].sel(wavelengths=850, method="nearest").isel(
            ortho_x=center_x, ortho_y=center_y
        )
        red = ds["reflectance"].sel(wavelengths=650, method="nearest").isel(
            ortho_x=center_x, ortho_y=center_y
        )
        ndvi = (nir - red) / (nir + red)
        ndvi_val = float(ndvi.values)
    except Exception:
        ndvi_val = 0.0

    print(f"📊 Pixel Analysis Complete. NDVI: {ndvi_val:.3f}")

    # Plotting
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, spectrum, "k-", linewidth=1.5, label="Surface Reflectance")

        # Annotation: Methane Trap
        plt.axvline(x=2300, color="red", linestyle="--", alpha=0.7, label="Methane (2300nm)")
        plt.text(2310, spectrum.max(), "CH4 Abs.", color="red", rotation=90)

        # Annotation: Clay/Lithium
        plt.axvline(x=2200, color="blue", linestyle="--", alpha=0.5, label="Clay/Li (2200nm)")

        # Annotation: Red Edge (Vegetation)
        plt.axvspan(680, 750, color="green", alpha=0.1, label="Red Edge (Veg)")

        plt.title(
            f"Spectral Fingerprint (Lat: {ds.latitude.values[center_y, center_x]:.4f})\nNDVI: {ndvi_val:.2f}"
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.show()
    else:
        print("ℹ️ Plotting disabled (headless mode).")


# --- CLI DRIVER ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chemical Eye: Orbital Hyper-Spectral Analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: methane
    parser_methane = subparsers.add_parser("methane", help="Detect Methane Plumes (L2B)")
    parser_methane.add_argument("--bbox", type=str, required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser_methane.add_argument("--start", type=str, default="2023-05-01", help="YYYY-MM-DD")
    parser_methane.add_argument("--end", type=str, default="2023-08-30", help="YYYY-MM-DD")
    parser_methane.add_argument(
        "--webhook", type=str, default=None, help="Optional webhook URL to post detection JSON."
    )
    parser_methane.add_argument(
        "--no-plot", action="store_true", help="Disable plume plotting (for headless/automation)."
    )

    # Command: spectrum
    parser_spec = subparsers.add_parser("spectrum", help="Analyze Surface Chemistry (L2A)")
    parser_spec.add_argument("--bbox", type=str, required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser_spec.add_argument("--start", type=str, default="2023-05-01", help="YYYY-MM-DD")
    parser_spec.add_argument("--end", type=str, default="2023-05-30", help="YYYY-MM-DD")
    parser_spec.add_argument("--no-plot", action="store_true", help="Disable plotting (for headless/automation).")

    args = parser.parse_args()

    # Parse Bbox
    try:
        bbox_tuple = tuple(map(float, args.bbox.split(",")))
    except ValueError:
        print("❌ Error: bbox must be 'min_lon,min_lat,max_lon,max_lat'")
        sys.exit(1)

    setup_auth()

    if args.command == "methane":
        result = run_methane_hunter(
            bbox_tuple,
            (args.start, args.end),
            plot=not args.no_plot,
            webhook=args.webhook,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "spectrum":
        run_spectral_analysis(bbox_tuple, (args.start, args.end), plot=not args.no_plot)
