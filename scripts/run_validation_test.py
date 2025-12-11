"""
Backtesting harness for Chemical Eye methane detection against ground truth.

Uses a list of TargetEvents (lat, lon, date, emission_rate_kg_hr) produced by
ingest_permian_truth.py and compares them against detections from the
MethaneDetector pipeline using EMIT L2B products.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

import earthaccess  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemeye.services.methane import MethaneDetector  # noqa: E402


@dataclass
class TargetEvent:
    lat: float
    lon: float
    date: str  # YYYY-MM-DD
    emission_rate_kg_hr: float | None = None


def load_events(path: Path) -> List[TargetEvent]:
    with path.open() as f:
        raw = json.load(f)
    events: list[TargetEvent] = []
    for item in raw:
        events.append(
            TargetEvent(
                lat=float(item["lat"]),
                lon=float(item["lon"]),
                date=str(item["date"])[:10],
                emission_rate_kg_hr=item.get("emission_rate_kg_hr"),
            )
        )
    return events


def detection_hit(
    event: TargetEvent,
    detections: list,
    max_distance_deg: float = 0.005,  # ~500m
) -> bool:
    for d in detections:
        d_lat = float(d.lat)
        d_lon = float(d.lon)
        if abs(d_lat - event.lat) <= max_distance_deg and abs(d_lon - event.lon) <= max_distance_deg:
            return True
    return False


def run_backtest(events: List[TargetEvent]) -> dict:
    detector = MethaneDetector()

    total_targets = len(events)
    emit_overlaps = 0
    matched = 0

    for idx, event in enumerate(events, 1):
        print(f"[{idx}/{total_targets}] Testing event at ({event.lat:.4f}, {event.lon:.4f}) on {event.date}...")

        bbox = (
            event.lon - 0.2,
            event.lat - 0.2,
            event.lon + 0.2,
            event.lat + 0.2,
        )

        try:
            result = detector.detect(
                bbox=bbox,
                start_date=event.date,
                end_date=event.date,
                max_granules=5,
            )
        except Exception as e:
            print(f"  ❌ Detection error: {e}")
            continue

        if result.scanned_count == 0:
            print("  ℹ️ No EMIT L2B coverage on this date.")
            continue

        emit_overlaps += 1

        if detection_hit(event, result.detections):
            matched += 1
            print("  ✅ HIT: plume detected near ground-truth location.")
        else:
            print("  ✖️ MISS: no plume detected near this event.")

    sensitivity = float(matched) / float(emit_overlaps) if emit_overlaps > 0 else 0.0

    return {
        "total_targets": total_targets,
        "emit_overlaps": emit_overlaps,
        "detections_matched": matched,
        "sensitivity": sensitivity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest against PermianMAP truth events")
    parser.add_argument(
        "--truth-file",
        type=str,
        default="permian_truth_events.json",
        help="Path to JSON produced by ingest_permian_truth.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    # Authenticate once
    print("🔐 Authenticating with NASA EarthData...")
    earthaccess.login()

    events = load_events(Path(args.truth_file))
    report = run_backtest(events)

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("\nValidation report:")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()

