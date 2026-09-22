"""Inspect raw recording integrity and measured timing without opening devices."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def inspect(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    with (path / "data.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (path / "debug_timing.csv").open() as f:
        frames = list(csv.DictReader(f))
    controls = [json.loads(line) for line in (path / "controls.jsonl").read_text().splitlines()]
    sides = tuple(metadata.get("camera_sides", ("left", "right")))
    missing = [r[f"{side}_image"] for r in rows for side in sides
               if r[f"{side}_image"] and not (path / r[f"{side}_image"]).is_file()]
    valid = [r for r in rows if r["valid"] == "1"]
    def stats(values):
        return dict(zip(("min", "median", "p95", "max"), map(float, np.percentile(values, [0, 50, 95, 100])))) if values else {}
    def hz(times):
        return (len(times)-1)*1e9/(max(times)-min(times)) if len(times)>1 and max(times)>min(times) else 0.
    stamps = [c["timestamp_ns"] for c in controls]
    return {
        "status": metadata["status"], "errors": metadata["errors"], "missing_images": missing,
        "raw_captures": len(rows), "valid_training_rows": len(valid),
        "rejected_captures": len(rows)-len(valid), "raw_control_ticks": len(controls),
        "control_hz": hz(stamps),
        "control_period_ms": stats(list(np.diff(stamps)/1e6)),
        "camera_hz": {side: hz([int(f["zed_timestamp_ns"]) for f in frames if f["side"] == side])
                      for side in sides},
        "camera_to_control_skew_ms": stats([int(r["skew_ns"])/1e6 for r in valid]),
        "pair_timestamp_skew_ms": stats([int(r["pair_skew_ns"])/1e6 for r in rows if r["pair_skew_ns"]]),
        "image_delivery_ms": stats([(int(f["host_monotonic_ns"])-int(f["image_monotonic_ns"]))/1e6 for f in frames]),
        "arm_read_span_ms": stats([(max(b for a,b in c["read_intervals"])-min(a for a,b in c["read_intervals"]))/1e6
                                  for c in controls if c["read_intervals"]]),
        "encoder_cache_age": "unknown; read intervals are software timestamps",
        "max_writer_queue": metadata.get("max_writer_queue"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    result = inspect(args.episode)
    print(json.dumps(result, indent=2))
    if result["status"] != "complete" or result["missing_images"] or not result["valid_training_rows"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
