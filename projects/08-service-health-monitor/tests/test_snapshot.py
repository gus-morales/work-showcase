"""Integration test on the real (deterministic, seeded) synthetic
panel: each of the four injected anomalies should produce an alert
event, and the sustained error-rate shift should produce exactly one,
not one per day it stays elevated."""
from generate_data import make_service_metrics
from snapshot import build_snapshot, default_detector_map


def test_all_four_injected_anomalies_are_caught():
    df = make_service_metrics()
    metrics = [c for c in df.columns if c != "day"]
    snapshot = build_snapshot(df, default_detector_map(metrics))
    events_by_metric = {}
    for event in snapshot["events"]:
        events_by_metric.setdefault(event["metric"], []).append(event)

    assert any(e["day"] == 45 for e in events_by_metric.get("p95_latency_ms", []))
    assert any(e["day"] == 55 and "data_gap" in e["detectors"] for e in events_by_metric.get("queue_depth", []))
    assert any(e["day"] == 75 for e in events_by_metric.get("error_rate", []))
    assert any(e["day"] == 105 for e in events_by_metric.get("request_volume", []))


def test_sustained_error_rate_shift_produces_exactly_one_event():
    df = make_service_metrics()
    metrics = [c for c in df.columns if c != "day"]
    snapshot = build_snapshot(df, default_detector_map(metrics))
    error_rate_events = [e for e in snapshot["events"] if e["metric"] == "error_rate"]
    assert len(error_rate_events) == 1
    assert error_rate_events[0]["day"] == 75
