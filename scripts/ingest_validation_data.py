"""
Ingest PermianMAP / Carbon Mapper validation data for backtesting.

Usage:
    python scripts/ingest_validation_data.py --input path/to/permian.geojson --output validation_truth_events.json --min-rate 100

Reads a GeoJSON/CSV containing plume detections, filters by emission rate
threshold (kg/hr), and writes a JSON list of {lat, lon, date, emission_rate_kg_hr}.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import geopandas as gpd


@dataclass
class TruthEvent:
    lat: float
    lon: float
    date: str  # YYYY-MM-DD
    emission_rate_kg_hr: float | None


def load_and_filter(path: Path, min_rate: float) -> List[TruthEvent]:
    gdf = gpd.read_file(path)

    if gdf.geometry is not None and not gdf.geometry.is_empty.all():
        gdf = gdf.to_crs("EPSG:4326")
        gdf["lat"] = gdf.geometry.y
        gdf["lon"] = gdf.geometry.x
    elif {"lat", "lon"}.issubset(gdf.columns):
        pass
    else:
        raise ValueError("Input must have Point geometry or lat/lon columns")

    # date column
    date_col = None
    for c in ("date", "observation_date", "acq_date"):
        if c in gdf.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("Could not find a date column (date / observation_date / acq_date)")

    # emission rate column (optional)
    emission_col = None
    for c in ("emission_rate_kg_hr", "emission_kg_hr", "emission_rate"):
        if c in gdf.columns:
            emission_col = c
            break

    events: list[TruthEvent] = []
    for _, row in gdf.iterrows():
        rate = float(row[emission_col]) if emission_col and row[emission_col] is not None else None
        if rate is not None and rate < min_rate:
            continue
        events.append(
            TruthEvent(
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                date=str(row[date_col])[:10],
                emission_rate_kg_hr=rate,
            )
        )
    return events


def main():
    parser = argparse.ArgumentParser(description="Ingest PermianMAP validation data")
    parser.add_argument("--input", required=True, help="Path to GeoJSON/CSV file")
    parser.add_argument(
        "--output",
        default="validation_truth_events.json",
        help="Path to write filtered truth events",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=100.0,
        help="Minimum emission rate (kg/hr) to include",
    )
    args = parser.parse_args()

    events = load_and_filter(Path(args.input), args.min_rate)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([asdict(e) for e in events], f, indent=2)
    print(f"Wrote {len(events)} events to {out_path}")


if __name__ == "__main__":
    main()
