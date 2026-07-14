"""
The anomaly engine: four detector types behind one interface, each
built to catch a different shape of anomaly that the others miss. A
detector is anything with a `.name` and a `.detect(series)` that
returns a boolean flag per day; adding a fifth kind is a one-class
addition here, nothing else in the pipeline needs to change.

- ThresholdDetector: any value past a fixed limit, regardless of
  history. Right for hard SLAs, wrong for metrics whose normal range
  drifts over time.
- ZScoreDetector: an outlier relative to a trailing rolling window.
  Catches a sudden spike a fixed threshold would miss if the normal
  range varies; blind to a slow drift, since the rolling window
  eventually absorbs it as the new normal.
- TrendBreakDetector: a sustained move between a short window and a
  longer baseline window. Catches a gradual decline while it's still
  developing, the case a threshold or a short z-score window misses.
- DataGapDetector: a day the metric didn't report at all. A pipeline
  or instrumentation outage, not a bad value.

Running all four against every metric (see snapshot.py) is the point:
no single method covers every anomaly shape a real service produces.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Detector(ABC):
    name: str

    @abstractmethod
    def detect(self, series: pd.Series) -> pd.Series:
        """Boolean flag per day, same index as series."""


class ThresholdDetector(Detector):
    name = "threshold"

    def __init__(self, limit: float, direction: str = "above"):
        if direction not in ("above", "below"):
            raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
        self.limit = limit
        self.direction = direction

    def detect(self, series: pd.Series) -> pd.Series:
        flags = series > self.limit if self.direction == "above" else series < self.limit
        return flags.fillna(False)


class ZScoreDetector(Detector):
    name = "zscore"

    def __init__(self, window: int = 14, z_thresh: float = 3.0, min_periods: int = 7):
        self.window = window
        self.z_thresh = z_thresh
        self.min_periods = min_periods

    def detect(self, series: pd.Series) -> pd.Series:
        rolling_mean = series.rolling(self.window, min_periods=self.min_periods).mean()
        rolling_std = series.rolling(self.window, min_periods=self.min_periods).std()
        z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        return (z.abs() > self.z_thresh).fillna(False)


class TrendBreakDetector(Detector):
    name = "trend_break"

    def __init__(self, short_window: int = 7, baseline_window: int = 30, rel_thresh: float = 0.15):
        self.short_window = short_window
        self.baseline_window = baseline_window
        self.rel_thresh = rel_thresh

    def detect(self, series: pd.Series) -> pd.Series:
        short_avg = series.rolling(self.short_window, min_periods=self.short_window).mean()
        baseline_avg = (
            series.rolling(self.baseline_window, min_periods=self.baseline_window)
            .mean()
            .shift(self.short_window)
        )
        rel_change = (short_avg - baseline_avg) / baseline_avg
        return (rel_change.abs() > self.rel_thresh).fillna(False)


class DataGapDetector(Detector):
    name = "data_gap"

    def detect(self, series: pd.Series) -> pd.Series:
        return series.isna()


def run_detectors(df: pd.DataFrame, detector_map: dict[str, list[Detector]]) -> pd.DataFrame:
    """Run each metric's assigned detectors over its daily series.
    Returns long-form: day, metric, detector, flagged (one row per
    metric-detector-day)."""
    rows = []
    for metric, detectors in detector_map.items():
        series = df.set_index("day")[metric]
        for detector in detectors:
            flags = detector.detect(series)
            for day, flagged in flags.items():
                rows.append({"day": int(day), "metric": metric, "detector": detector.name, "flagged": bool(flagged)})
    return pd.DataFrame(rows, columns=["day", "metric", "detector", "flagged"])
