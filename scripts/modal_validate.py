import modal
import sys
import os
from pathlib import Path

# Explicit imports to avoid module attribute errors
from modal import App, Image, Secret
from modal.mount import Mount

# Define Modal App (formerly Stub)
app = App("chemeye-algorithm-validator")

# Image with scientific dependencies
image = (
    Image.debian_slim()
    .pip_install(
        "earthaccess",
        "xarray[io]",
        "h5netcdf",
        "scipy",
        "matplotlib",
        "numpy",
        "rioxarray"
    )
    .env({"EARTHDATA_USERNAME": "", "EARTHDATA_PASSWORD": ""}) 
)

# Mount the source code
src_mount = Mount.from_local_dir(
    Path(__file__).parent.parent / "src",
    remote_path="/root/src"
)

# Mount the validation script logic
script_mount = Mount.from_local_file(
    Path(__file__).parent / "validate_algorithm.py",
    remote_path="/root/validate_algorithm.py"
)

@app.function(
    image=image,
    secrets=[Secret.from_name("nasa-earthdata")],
    mounts=[src_mount, script_mount],
    timeout=1800,
    memory=4096,
    gpu=None
)
def run_remote_validation():
    import sys
    import shutil
    
    # Ensure standard paths
    sys.path.append("/root")
    sys.path.append("/root/src")
    
    print("🚀 Launched methane AMF validation on Modal Cloud Runner")
    
    # Import validation logic
    try:
        from validate_algorithm import validate_algorithm
    except ImportError as e:
        print(f"Import failed: {e}")
        # Debugging: List files
        import os
        print(f"Root contents: {os.listdir('/root')}")
        print(f"Src contents: {os.listdir('/root/src')}")
        return {"error": str(e)}

    # Output directory
    output_dir = "/tmp/algorithm_results"
    
    # Run validation
    try:
        validate_algorithm(z_threshold=4.0, save_dir=output_dir)
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
    results = {}
    
    # Read JSON results
    import json
    try:
        with open(f"{output_dir}/algorithm_results.json", "r") as f:
            results["data"] = json.load(f)
    except FileNotFoundError:
        results["data"] = {"error": "No results JSON found"}
        
    # Read Image
    try:
        with open(f"{output_dir}/amf_validation.png", "rb") as f:
            results["image"] = f.read()
    except FileNotFoundError:
        results["image"] = None
        
    return results

@app.local_entrypoint()
def main():
    print("✨ Triggering methane AMF validation on Modal...")
    try:
        results = run_remote_validation.remote()
    except Exception as e:
        print(f"Remote execution failed: {e}")
        return

    if "error" in results:
        print(f"❌ Remote error: {results['error']}")
        return

    if results.get("image"):
        local_img = "amf_validation_cloud.png"
        with open(local_img, "wb") as f:
            f.write(results["image"])
        print(f"✅ Downloaded validation plot to {local_img}")
    
    if "data" in results:
        import json
        print(json.dumps(results["data"], indent=2))
