"""
The pipeline: load the catalog, pull both panels from DuckDB via the
sql/ queries, run each engine against the metrics it owns, and write
one unified snapshot. Nothing downstream, alerting, the dashboard, the
notebook, ever recomputes a detector, runs popmon, or queries the
database directly; it reads snapshot/flags.csv and snapshot/events.json.

Run:
    python src/snapshot.py
"""
from pathlib import Path

import duckdb
import pandas as pd

from alerts import ConsoleChannel, FileChannel, dispatch, find_alert_events
from catalog import load_catalog, model_features, ops_metrics
from detectors import DataGapDetector, ThresholdDetector, TrendBreakDetector, ZScoreDetector, run_detectors
from stability import extract_alerts, run_stability_report

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
SQL_DIR = BASE / "sql"
CATALOG_DIR = BASE / "catalog"
SNAPSHOT_DIR = BASE / "snapshot"
DB_PATH = DATA_DIR / "observatory.duckdb"
START_DATE = pd.Timestamp("2026-01-01")


def _run_sql(con, filename: str) -> pd.DataFrame:
    return con.execute((SQL_DIR / filename).read_text()).fetchdf()


def ops_detector_map(entries: list) -> dict[str, list]:
    return {
        entry.name: [
            ThresholdDetector(entry.threshold_limit, entry.threshold_direction),
            ZScoreDetector(),
            TrendBreakDetector(),
            DataGapDetector(),
        ]
        for entry in entries
    }


def build_snapshot(ops_df: pd.DataFrame, scoring_df: pd.DataFrame, catalog_entries: list) -> dict:
    ops_entries = ops_metrics(catalog_entries)
    feature_entries = model_features(catalog_entries)

    ops_flags = run_detectors(ops_df, ops_detector_map(ops_entries))

    feature_names = [entry.name for entry in feature_entries]
    report = run_stability_report(scoring_df, time_axis="date", features=feature_names)
    model_flags = extract_alerts(report, START_DATE)

    flags = pd.concat([ops_flags, model_flags], ignore_index=True)
    events = find_alert_events(flags)
    return {"flags": flags, "events": events, "stability_report": report}


def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    catalog_entries = load_catalog(CATALOG_DIR)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    ops_df = _run_sql(con, "ops_daily_panel.sql")
    scoring_df = _run_sql(con, "scoring_daily_panel.sql")
    con.close()

    snapshot = build_snapshot(ops_df, scoring_df, catalog_entries)
    snapshot["flags"].to_csv(SNAPSHOT_DIR / "flags.csv", index=False)

    events_df = pd.DataFrame(snapshot["events"])
    events_df.to_json(SNAPSHOT_DIR / "events.json", orient="records", indent=2)

    snapshot["stability_report"].to_file(str(SNAPSHOT_DIR / "popmon_stability_report.html"))

    print(f"Wrote {len(snapshot['flags'])} flag rows -> snapshot/flags.csv")
    print(f"Wrote {len(snapshot['events'])} alert event(s) -> snapshot/events.json")
    print("Wrote snapshot/popmon_stability_report.html")

    log_path = SNAPSHOT_DIR / "alert_log.jsonl"
    log_path.write_text("")
    channels = [ConsoleChannel(), FileChannel(log_path)]
    dispatch(snapshot["events"], channels)


if __name__ == "__main__":
    main()
