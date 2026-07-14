"""
Synthetic data for project 08: an operational anomaly-monitoring
toolkit for a fictional web service's health metrics. Domain-agnostic
(latency, errors, traffic, queue depth), not tied to the fintech
company in projects 01-03/05, since this project is about the
monitoring architecture (pluggable detectors, pluggable alert
channels, snapshot-based decoupling), not about any one company's
metrics.

One dataset: service_metrics.csv, a daily panel of four metrics, each
with a deliberately different kind of injected anomaly, chosen so each
one needs a different detector to catch cleanly:

- p95_latency_ms: a short spike (days 45-47) - a threshold breach.
- error_rate: a sustained step change (day 75 on) - a statistical
  outlier that a fixed threshold could also catch, but which should
  only trigger one alert, not one per day it stays elevated.
- request_volume: a gradual decline (days 95-109, settling ~30% lower)
  - a trend break, the kind a static threshold or a short rolling
  window would miss while it's still developing.
- queue_depth: a reporting gap (days 55-57) - missing data, not a bad
  value.

Run:
    python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 11
rng = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

N_DAYS = 120


def make_service_metrics(n_days: int = N_DAYS) -> pd.DataFrame:
    days = np.arange(n_days)

    p95_latency_ms = rng.normal(180, 15, size=n_days)
    p95_latency_ms[45:48] = rng.normal(600, 40, size=3)

    error_rate = rng.normal(0.015, 0.0015, size=n_days)
    error_rate[75:] = rng.normal(0.06, 0.003, size=n_days - 75)

    request_volume = rng.normal(1200, 80, size=n_days).astype(float)
    request_volume[95:110] *= 1 - np.linspace(0, 0.30, 15)
    request_volume[110:] *= 0.70

    queue_depth = rng.normal(40, 8, size=n_days)

    df = pd.DataFrame({
        "day": days,
        "p95_latency_ms": p95_latency_ms.round(1),
        "error_rate": error_rate.clip(0, 1).round(4),
        "request_volume": request_volume.round(0).astype(int),
        "queue_depth": queue_depth.round(1),
    })

    # queue_depth didn't report for 3 days (pipeline outage); the other
    # metrics still did, so it's a gap in one column, not a dropped day.
    df.loc[55:57, "queue_depth"] = np.nan

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_service_metrics()
    df.to_csv(OUT_DIR / "service_metrics.csv", index=False)
    print(f"Wrote {len(df)} daily rows -> data/service_metrics.csv")
    print("Injected: latency spike (days 45-47), error-rate step change (day 75+), "
          "request-volume decline (days 95-109), queue-depth reporting gap (days 55-57)")


if __name__ == "__main__":
    main()
