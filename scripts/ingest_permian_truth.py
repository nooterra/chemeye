"""
Ingest PermianMAP / Carbon Mapper ground-truth plume detections.

This script reads a local CSV or GeoJSON file containing methane plume
detections (lat, lon, date, emission_rate_kg_hr) and converts it into
an internal JSON format used by the validation harness.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import geopandas as gpd


@dataclass
class TargetEvent:
    lat: float
    lon: float
    date: str  # ISO date string YYYY-MM-DD
    emission_rate_kg_hr: float | None = None


def load_truth_file(path: Path) -> List[TargetEvent]:
    """
    Load a GeoJSON/CSV file of plume detections into TargetEvent objects.

    This function assumes the source file has at least:
      - geometry with Point coordinates (for GeoJSON)
      - or numeric 'lat'/'lon' columns (for CSV)
      - a date column named one of: 'date', 'observation_date'
      - an emission column named one of: 'emission_rate_kg_hr', 'emission_kg_hr'
    """
    gdf = gpd.read_file(path)

    # Ensure we have latitude/longitude
    if gdf.geometry is not None and not gdf.geometry.is_empty.all():
        gdf = gdf.to_crs("EPSG:4326")
        gdf["lat"] = gdf.geometry.y
        gdf["lon"] = gdf.geometry.x
    elif {"lat", "lon"}.issubset(gdf.columns):
        # Already has lat/lon columns
        pass
    else:
        raise ValueError("Input file must have Point geometry or 'lat'/'lon' columns")

    # Date column heuristics
    date_col = None
    for candidate in ("date", "observation_date", "acq_date"):
        if candidate in gdf.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("Could not find a date column (expected: date, observation_date, acq_date)")

    # Emission rate (optional)
    emission_col = None
    for candidate in ("emission_rate_kg_hr", "emission_kg_hr"):
        if candidate in gdf.columns:
            emission_col = candidate
            break

    events: list[TargetEvent] = []
    for _, row in gdf.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        date_str = str(row[date_col])[:10]  # normalize to YYYY-MM-DD
        emission = float(row[emission_col]) if emission_col and row[emission_col] is not None else None

        events.append(TargetEvent(lat=lat, lon=lon, date=date_str, emission_rate_kg_hr=emission))

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PermianMAP / Carbon Mapper truth data")
    parser.add_argument("input", type=str, help="Path to local GeoJSON/CSV truth dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="permian_truth_events.json",
        help="Output JSON file containing TargetEvent list",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    events = load_truth_file(input_path)

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump([asdict(e) for e in events], f, indent=2)

    print(f"Wrote {len(events)} target events to {output_path}")


if __name__ == "__main__":
    main()

