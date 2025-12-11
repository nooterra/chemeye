
import earthaccess
import os
from dotenv import load_dotenv

load_dotenv()

bbox = (-106, 29, -99, 35)
print("Searching for granules in Permian Basin (Aug 2023)...")

results = earthaccess.search_data(
    short_name="EMITL2ARFL",
    temporal=("2023-08-01", "2023-08-10"),
    bounding_box=bbox,
    count=20
)

print(f"Found {len(results)} granules.")
for g in results:
    print(f" - {g['umm']['GranuleUR']}")
