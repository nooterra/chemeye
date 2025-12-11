"""
Backtester for Chemical Eye AMF detection against ground truth events.

Usage:
    python scripts/run_backtest.py --truth-file validation_truth_events.json --output validation_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

import earthaccess  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemeye.services.methane_amf import run_matched_filter, extract_plume_locations  # noqa: E402
from chemeye.services.emit import get_emit_service  # noqa: E402


@dataclass
class TruthEvent:
    lat: float
    lon: float
    date: str
    emission_rate_kg_hr: float | None


def load_truth(path: Path) -> List[TruthEvent]:
    with path.open() as f:
        raw = json.load(f)
    events: list[TruthEvent] = []
    for item in raw:
        events.append(
            TruthEvent(
                lat=float(item["lat"]),
                lon=float(item["lon"]),
                date=str(item["date"])[:10],
                emission_rate_kg_hr=item.get("emission_rate_kg_hr"),
            )
        )
    return events


def detection_hit(event: TruthEvent, plumes: list, max_distance_deg: float = 0.005) -> bool:
    for p in plumes:
        d_lat = float(p.lat)
        d_lon = float(p.lon)
        if abs(d_lat - event.lat) <= max_distance_deg and abs(d_lon - event.lon) <= max_distance_deg:
            return True
    return False


def run_backtest(events: List[TruthEvent]) -> dict:
    emit_service = get_emit_service()
    emit_service.authenticate()

    total_targets = len(events)
    emit_overlaps = 0
    matched = 0

    for idx, event in enumerate(events, 1):
        print(f"[{idx}/{total_targets}] Event at ({event.lat:.4f}, {event.lon:.4f}) on {event.date}")

        bbox = (
            event.lon - 0.2,
            event.lat - 0.2,
            event.lon + 0.2,
            event.lat + 0.2,
        )

        # Find L2A granules overlapping location/date
        try:
            granules = earthaccess.search_data(
                short_name="EMITL2ARFL",
                temporal=(event.date, event.date),
                bounding_box=bbox,
                count=1,
            )
        except Exception as e:
            print(f"  ❌ Search failed: {e}")
            continue

        if not granules:
            print("  ℹ️ No EMIT L2A coverage on this date.")
            continue

        emit_overlaps += 1

        try:
            files = emit_service.open_granules([granules[0]])
            if not files:
                print("  ❌ Could not open granule")
                continue
            ds = emit_service.load_dataset(files[0])
            if ds is None:
                print("  ❌ Failed to load dataset")
                continue
        except Exception as e:
            print(f"  ❌ Load failed: {e}")
            continue

        try:
            z_map, metadata = run_matched_filter(
                ds,
                z_threshold=3.0,
                mode="cluster",
                n_clusters=10,
                iterative_background=True,
            )
        except Exception as e:
            print(f"  ❌ AMF failed: {e}")
            continue

        plumes = []
        if "lat" in ds and "lon" in ds:
            lat_array = ds["lat"].values
            lon_array = ds["lon"].values
            plumes = extract_plume_locations(z_map, lat_array, lon_array, min_pixels=5)

        if detection_hit(event, plumes):
            matched += 1
            print("  ✅ HIT")
        else:
            print("  ✖️ MISS")

    sensitivity = float(matched) / float(emit_overlaps) if emit_overlaps > 0 else 0.0
    false_negative_rate = 1.0 - sensitivity if emit_overlaps > 0 else 1.0

    return {
        "total_targets": total_targets,
        "emit_overlaps": emit_overlaps,
        "detections_matched": matched,
        "sensitivity": sensitivity,
        "false_negative_rate": false_negative_rate,
        "detection_limit_kg_hr": 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest Chemical Eye against ground truth")
    parser.add_argument(
        "--truth-file",
        default="validation_truth_events.json",
        help="JSON produced by ingest_validation_data.py",
    )
    parser.add_argument(
        "--output",
        default="validation_report.json",
        help="Where to write the report JSON",
    )
    args = parser.parse_args()

    print("🔐 Authenticating with NASA EarthData...")
    earthaccess.login()

    events = load_truth(Path(args.truth_file))
    report = run_backtest(events)

    out_path = Path(args.output)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("\nValidation report:")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
