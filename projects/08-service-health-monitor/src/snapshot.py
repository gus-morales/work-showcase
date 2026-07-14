"""
The pipeline: run the full detector engine over the metrics panel once,
write the result to snapshot/, and dispatch alerts for whatever's newly
flagged. Nothing downstream (the notebook, a dashboard, a report) ever
recomputes this; it just reads flags.csv and events.json. That
decoupling is the actual point of a snapshot architecture: the
detection run is reproducible and inspectable on its own, and a slow
or broken data source can't take the reporting layer down with it.

Run:
    python src/snapshot.py
"""
from pathlib import Path

import pandas as pd

from alerts import ConsoleChannel, FileChannel, dispatch, find_alert_events
from detectors import DataGapDetector, ThresholdDetector, TrendBreakDetector, ZScoreDetector, run_detectors

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
SNAPSHOT_DIR = BASE / "snapshot"

# Every metric gets the full detector stack; which one(s) actually
# fire is what tells you the shape of the anomaly, not just that
# something's wrong.
THRESHOLDS = {
    "p95_latency_ms": (450, "above"),
    "error_rate": (0.05, "above"),
    "request_volume": (400, "below"),
    "queue_depth": (150, "above"),
}


def default_detector_map(metrics: list[str]) -> dict[str, list]:
    return {
        metric: [
            ThresholdDetector(*THRESHOLDS[metric]),
            ZScoreDetector(),
            TrendBreakDetector(),
            DataGapDetector(),
        ]
        for metric in metrics
    }


def build_snapshot(df: pd.DataFrame, detector_map: dict[str, list]) -> dict:
    flags = run_detectors(df, detector_map)
    events = find_alert_events(flags)
    return {"flags": flags, "events": events}


def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / "service_metrics.csv")
    metrics = [c for c in df.columns if c != "day"]

    snapshot = build_snapshot(df, default_detector_map(metrics))
    snapshot["flags"].to_csv(SNAPSHOT_DIR / "flags.csv", index=False)

    events_df = pd.DataFrame(snapshot["events"])
    events_df.to_json(SNAPSHOT_DIR / "events.json", orient="records", indent=2)

    print(f"Wrote {len(snapshot['flags'])} flag rows -> snapshot/flags.csv")
    print(f"Wrote {len(snapshot['events'])} alert event(s) -> snapshot/events.json")

    log_path = SNAPSHOT_DIR / "alert_log.jsonl"
    log_path.write_text("")  # fresh log each run, so re-running is idempotent to inspect
    channels = [ConsoleChannel(), FileChannel(log_path)]
    dispatch(snapshot["events"], channels)


if __name__ == "__main__":
    main()
