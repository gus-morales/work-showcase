"""Unit tests for extract_alerts: a feature's own reference-period
red-check baseline sets what counts as elevated, not a flat 'any red'
rule, and the reference window itself is never flagged."""
from types import SimpleNamespace

import pandas as pd

from stability import extract_alerts

START = pd.Timestamp("2026-01-01")


def _counts(n_red_by_day: dict) -> pd.DataFrame:
    days = sorted(n_red_by_day)
    index = [START + pd.Timedelta(days=d) + pd.Timedelta(hours=12) for d in days]
    return pd.DataFrame({"n_red": [n_red_by_day[d] for d in days]}, index=index)


def _report(alerts: dict) -> SimpleNamespace:
    return SimpleNamespace(datastore={"alerts": alerts})


def test_sustained_elevation_above_baseline_is_flagged():
    n_red = {d: 1 for d in range(15)}
    n_red.update({d: 5 for d in range(15, 25)})
    report = _report({"feat": _counts(n_red)})

    result = extract_alerts(report, START, reference_days=range(0, 15), red_margin=2)
    flagged_days = set(result[result["flagged"]]["day"])
    assert flagged_days == set(range(15, 25))


def test_reference_window_itself_is_never_flagged():
    n_red = {d: 10 for d in range(15)}
    report = _report({"feat": _counts(n_red)})

    result = extract_alerts(report, START, reference_days=range(0, 15), red_margin=2)
    assert not result["flagged"].any()


def test_background_noise_within_margin_is_not_flagged():
    n_red = {d: 1 for d in range(15)}
    n_red.update({d: 2 for d in range(15, 30)})
    report = _report({"feat": _counts(n_red)})

    result = extract_alerts(report, START, reference_days=range(0, 15), red_margin=2)
    assert not result["flagged"].any()


def test_aggregate_pseudo_feature_is_skipped():
    n_red = {d: 10 for d in range(20)}
    report = _report({"_AGGREGATE_": _counts(n_red), "feat": _counts({d: 0 for d in range(20)})})

    result = extract_alerts(report, START, reference_days=range(0, 15), red_margin=2)
    assert set(result["metric"]) == {"feat"}
