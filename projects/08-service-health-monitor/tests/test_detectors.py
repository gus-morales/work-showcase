"""Unit tests for each detector type on small hand-built series: each
one should catch the anomaly shape it's built for and stay quiet on
values it isn't."""
import numpy as np
import pandas as pd

from detectors import DataGapDetector, ThresholdDetector, TrendBreakDetector, ZScoreDetector


def test_threshold_detector_flags_above():
    series = pd.Series([10, 20, 30, 500, 25])
    flags = ThresholdDetector(limit=100, direction="above").detect(series)
    assert list(flags) == [False, False, False, True, False]


def test_threshold_detector_flags_below():
    series = pd.Series([1200, 1180, 300, 1210])
    flags = ThresholdDetector(limit=500, direction="below").detect(series)
    assert list(flags) == [False, False, True, False]


def test_threshold_detector_rejects_bad_direction():
    try:
        ThresholdDetector(limit=1, direction="sideways")
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_zscore_detector_flags_a_spike_against_a_flat_baseline():
    values = [180.0] * 20 + [600.0] + [180.0] * 10
    series = pd.Series(values)
    flags = ZScoreDetector(window=14, z_thresh=3.0, min_periods=7).detect(series)
    assert flags.iloc[20]
    assert not flags.iloc[:20].any()


def test_zscore_detector_quiet_on_constant_series():
    series = pd.Series([100.0] * 30)
    flags = ZScoreDetector(window=14, min_periods=7).detect(series)
    assert not flags.any()


def test_trend_break_detector_catches_a_sustained_level_shift():
    baseline = [1000.0] * 40
    shifted = [700.0] * 20  # a 30% drop, well past the 15% default threshold
    series = pd.Series(baseline + shifted)
    flags = TrendBreakDetector(short_window=7, baseline_window=30, rel_thresh=0.15).detect(series)
    # once the short window is entirely inside the shifted period, it should flag
    assert flags.iloc[46:].any()
    # nothing should flag while the series is still flat
    assert not flags.iloc[:40].any()


def test_trend_break_detector_quiet_on_noise_within_threshold():
    rng = np.random.default_rng(0)
    series = pd.Series(1000 + rng.normal(0, 20, size=60))  # ~2% relative noise, well under 15%
    flags = TrendBreakDetector(short_window=7, baseline_window=30, rel_thresh=0.15).detect(series)
    assert not flags.any()


def test_data_gap_detector_flags_missing_values_only():
    series = pd.Series([1.0, 2.0, np.nan, 4.0, np.nan])
    flags = DataGapDetector().detect(series)
    assert list(flags) == [False, False, True, False, True]
