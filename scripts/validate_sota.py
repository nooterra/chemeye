"""
SOTA Validation Script

Compares SOTA Adaptive Matched Filter against NASA L2B product on known Permian Basin event.
This is the "Pepsi Challenge" - can we find what NASA found, plus more?
"""
import earthaccess
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import json
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemeye.services.sota_methane import run_matched_filter, extract_plume_locations


def find_permian_granule():
    """
    Search for the Golden Granule pair (Permian Super-Emitter, Aug 2023).
    """
    print("🔍 Searching for Golden Granule (Permian Super-Emitter)...")
    
    # Golden IDs provided by user (Aug 5 invalid, using Aug 4 confirmed as fallback)
    target_l2a = "EMIT_L2A_RFL_001_20230804T174129_2321612_015"
    target_l2b = "EMIT_L2B_CH4PLM_001_20230804T174129_2321612_015"
    
    # Search specific granules
    l2a_results = earthaccess.search_data(short_name="EMITL2ARFL", granule_name=target_l2a)
    l2b_results = earthaccess.search_data(short_name="EMITL2BCH4PLM", granule_name=target_l2b)
    
    if l2a_results and l2b_results:
        print(f"✓ Found exact requested granules: {target_l2a}")
        return l2a_results[0], l2b_results[0]
        
    print("⚠️  Golden granule not found in EarthData search results.")
    print("   Trying generic search + manual match...")
    
    # Fallback legacy search: date search (Aug 4, 2023)
    print("   Searching by date (2023-08-04) - where data is confirmed...")
    bbox = (-106, 29, -99, 35)
    l2a_results = earthaccess.search_data(
        short_name="EMITL2ARFL", 
        temporal=("2023-08-04T00:00:00", "2023-08-04T23:59:59"), 
        bounding_box=bbox
    )
    l2b_results = earthaccess.search_data(
        short_name="EMITL2BCH4PLM", 
        temporal=("2023-08-04T00:00:00", "2023-08-04T23:59:59"), 
        bounding_box=bbox
    )
    
    if not l2a_results or not l2b_results:
        raise ValueError("No granules found on 2023-08-04.")

    # Find matching granules (same orbit/time)
    l2a_names = {g['umm']['GranuleUR']: g for g in l2a_results}
    l2b_names = {g['umm']['GranuleUR']: g for g in l2b_results}
    
    # Extract orbit identifiers (format: EMIT_L2X_XXX_001_YYYYMMDDTHHMMSS_ORBITID_XXX)
    def get_orbit_id(name):
        parts = name.split('_')
        if len(parts) >= 6:
            # Timestamp (index 4) + Orbit (index 5)
            # e.g., EMIT_L2A_RFL_001_20230805T182042_2321712_012
            return parts[4] + '_' + parts[5]
        return None
    
    l2a_orbits = {get_orbit_id(name): name for name in l2a_names.keys()}
    l2b_orbits = {get_orbit_id(name): name for name in l2b_names.keys()}
    
    # Find common orbits
    common_orbits = set(l2a_orbits.keys()) & set(l2b_orbits.keys())
    
    if not common_orbits:
        raise ValueError(f"Found L2A/L2B data but no matching orbits.")
    
    # Pick the one closest to the requested time ("174129" roughly)
    orbit = sorted(list(common_orbits))[0] # Just pick first for now
    print(f"✓ Found matching granule pair via date search: {orbit}")
    
    return l2a_names[l2a_orbits[orbit]], l2b_names[l2b_orbits[orbit]]


def validate_sota(z_threshold=4.0, save_dir="./sota_results"):
    """
    Run SOTA validation comparing AMF to NASA L2B.
    
    Args:
        z_threshold: Detection threshold in standard deviations
        save_dir: Directory to save results
    """
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("SOTA METHANE DETECTION - VALIDATION")
    print("=" * 80)
    
    # 1. Authenticate
    print("\n🔐 Authenticating with NASA EarthData...")
    try:
        earthaccess.login()
    except Exception as e:
        print(f"❌Auth failed: {e}")
        print("Run: earthaccess.login(strategy='interactive') first")
        return
    
    # 2. Find granules
    try:
        l2a_granule, l2b_granule = find_permian_granule()
    except Exception as e:
        print(f"❌ Granule search failed: {e}")
        return
    
    l2a_name = l2a_granule['umm']['GranuleUR']
    l2b_name = l2b_granule['umm']['GranuleUR']
    
    print(f"\n📊 GRANULE PAIR:")
    print(f"  L2A (Raw):  {l2a_name}")
    print(f"  L2B (NASA): {l2b_name}")
    
    # 3. Download L2A (raw reflectance) - THIS IS HEAVY
    print(f"\n📥 Downloading L2A Reflectance (~300-500 MB)...")
    try:
        l2a_files = earthaccess.open([l2a_granule])
        print("  ✓ Opening NetCDF with xarray...")
        ds_l2a = xr.open_dataset(l2a_files[0], engine="h5netcdf", group=None)
    except Exception as e:
        print(f"❌ L2A download failed: {e}")
        return
    
    print(f"  ✓ Loaded: {ds_l2a.dims}")
    
    # 4. Run SOTA Matched Filter
    print(f"\n🔬 Running SOTA Adaptive Matched Filter (threshold={z_threshold}σ)...")
    try:
        sota_map, metadata = run_matched_filter(ds_l2a, z_threshold=z_threshold)
        print(f"  ✓ Complete!")
        print(f"    Max Z-Score: {metadata['max_z_score']:.2f}σ")
        print(f"    Detection Pixels: {metadata['n_detection_pixels']:,}")
    except Exception as e:
        print(f"❌ AMF failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Download L2B (NASA plume mask)
    print(f"\n📥 Downloading L2B Validation Data...")
    try:
        l2b_files = earthaccess.open([l2b_granule])
        ds_l2b = xr.open_dataset(l2b_files[0], engine="h5netcdf")
    except Exception as e:
        print(f"❌ L2B download failed: {e}")
        return
    
    # Extract NASA's plume mask
    if 'ch4_plume_complex' in ds_l2b:
        # Sum across plume dimension to get binary mask
        nasa_mask = ds_l2b['ch4_plume_complex'].sum(dim='number_of_plumes').values
        nasa_mask = np.nan_to_num(nasa_mask, nan=0)
    elif 'plume_complex' in ds_l2b:
        nasa_mask = ds_l2b['plume_complex'].values
    else:
        print(f"⚠️  L2B variables: {list(ds_l2b.data_vars)}")
        print("Creating empty NASA mask for comparison...")
        nasa_mask = np.zeros_like(sota_map)
    
    n_nasa_pixels = (nasa_mask > 0).sum()
    print(f"  ✓ NASA detected {n_nasa_pixels:,} plume pixels")
    
    # 6. Extract plume locations from SOTA
    if 'lat' in ds_l2a and 'lon' in ds_l2a:
        lat_array = ds_l2a['lat'].values
        lon_array = ds_l2a['lon'].values
        
        from chemeye.services.sota_methane import extract_plume_locations
        plumes = extract_plume_locations(sota_map, lat_array, lon_array, min_pixels=5)
        
        print(f"\n🎯 SOTA Plume Detections: {len(plumes)}")
        for i, p in enumerate(plumes[:10], 1):  # Top 10
            print(f"   {i}. ({p.lat:.4f}°, {p.lon:.4f}°) - {p.z_score:.1f}σ - {p.pixel_count} px")
    
    # 7. Visual Comparison
    print(f"\n📊 Generating Comparison Plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"SOTA vs NASA: {l2a_name}", fontsize=14)
    
    # Plot 1: SOTA Detection Map
    im1 = axes[0].imshow(sota_map, cmap='inferno', vmin=0, vmax=10)
    axes[0].set_title(f"SOTA: Adaptive Matched Filter\nMax: {metadata['max_z_score']:.1f}σ | Pixels: {metadata['n_detection_pixels']:,}")
    axes[0].set_xlabel("Crosstrack")
    axes[0].set_ylabel("Downtrack")
    plt.colorbar(im1, ax=axes[0], label="Z-Score (σ)")
    
    # Plot 2: NASA L2B Product
    im2 = axes[1].imshow(nasa_mask, cmap='Greys_r')
    axes[1].set_title(f"NASA L2B Product\nPixels: {n_nasa_pixels:,}")
    axes[1].set_xlabel("Crosstrack")
    plt.colorbar(im2, ax=axes[1], label="Plume Mask")
    
    # Plot 3: Overlay (SOTA in Red, NASA in Blue)
    overlay = np.zeros((sota_map.shape[0], sota_map.shape[1], 3))
    overlay[:,:,0] = np.clip(sota_map / 10, 0, 1)  # SOTA = Red
    overlay[:,:,2] = np.clip(nasa_mask / nasa_mask.max() if nasa_mask.max() > 0 else 0, 0, 1)  # NASA = Blue
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay\nRed=SOTA | Blue=NASA | Purple=Both")
    axes[2].set_xlabel("Crosstrack")
    
    plt.tight_layout()
    
    plot_path = Path(save_dir) / "sota_validation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {plot_path}")
    
    # 8. Save numerical results
    results = {
        "l2a_granule": l2a_name,
        "l2b_granule": l2b_name,
        "sota_metadata": metadata,
        "nasa_pixels": int(n_nasa_pixels),
        "z_threshold": z_threshold,
        "sota_plumes": len(plumes) if 'plumes' in locals() else 0
    }
    
    json_path = Path(save_dir) / "sota_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved to {json_path}")
    
    # 9. VERDICT
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if metadata['n_detection_pixels'] > n_nasa_pixels:
        print(f"🏆 SOTA FOUND MORE: {metadata['n_detection_pixels']:,} vs {n_nasa_pixels:,} pixels")
        print("   This suggests the AMF is detecting signals NASA filtering removed.")
    elif metadata['n_detection_pixels'] == n_nasa_pixels:
        print(f"✓ EXACT MATCH: {metadata['n_detection_pixels']:,} pixels")
    else:
        print(f"⚠️  SOTA MORE CONSERVATIVE: {metadata['n_detection_pixels']:,} vs {n_nasa_pixels:,}")
        print(f"   Try lowering threshold (currently {z_threshold}σ)")
    
    print("\nNext Steps:")
    print("  1. Inspect sota_validation.png visually")
    print("  2. If purple regions exist → validation successful")
    print("  3. If red-only regions exist → potential new discoveries")
    print("  4. Tune z_threshold for desired sensitivity")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate SOTA methane detection")
    parser.add_argument("--threshold", type=float, default=4.0, help="Z-score threshold")
    parser.add_argument("--output", type=str, default="./sota_results", help="Output directory")
    
    args = parser.parse_args()
    
    validate_sota(z_threshold=args.threshold, save_dir=args.output)
