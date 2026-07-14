"""Unit tests for alert dedup (find_alert_events) and channel dispatch:
a run of flagged days should produce exactly one event, on the day the
run starts, and every event should reach every configured channel."""
import pandas as pd

from alerts import AlertChannel, WebhookChannel, dispatch, find_alert_events


def _flags(rows):
    """rows: list of (day, metric, detector, flagged)."""
    return pd.DataFrame(rows, columns=["day", "metric", "detector", "flagged"])


def test_a_sustained_run_produces_exactly_one_event():
    rows = [(d, "error_rate", "threshold", d >= 5) for d in range(10)]
    events = find_alert_events(_flags(rows))
    assert len(events) == 1
    assert events[0]["day"] == 5
    assert events[0]["metric"] == "error_rate"


def test_a_run_that_resolves_and_recurs_produces_two_events():
    flagged = [True, True, False, False, True, True]
    rows = [(d, "queue_depth", "data_gap", f) for d, f in enumerate(flagged)]
    events = find_alert_events(_flags(rows))
    assert [e["day"] for e in events] == [0, 4]


def test_event_lists_every_detector_triggered_on_its_start_day():
    rows = [
        (0, "p95_latency_ms", "threshold", True),
        (0, "p95_latency_ms", "zscore", True),
        (0, "p95_latency_ms", "trend_break", False),
        (1, "p95_latency_ms", "threshold", True),
        (1, "p95_latency_ms", "zscore", True),
        (1, "p95_latency_ms", "trend_break", False),
    ]
    events = find_alert_events(_flags(rows))
    assert len(events) == 1
    assert events[0]["detectors"] == ["threshold", "zscore"]


def test_never_flagged_metric_produces_no_events():
    rows = [(d, "request_volume", "threshold", False) for d in range(10)]
    assert find_alert_events(_flags(rows)) == []


class _RecordingChannel(AlertChannel):
    def __init__(self):
        self.received = []

    def send(self, event):
        self.received.append(event)


def test_dispatch_sends_every_event_to_every_channel():
    events = [{"day": 1, "metric": "a", "detectors": ["threshold"]},
              {"day": 5, "metric": "b", "detectors": ["zscore"]}]
    channel_a, channel_b = _RecordingChannel(), _RecordingChannel()
    dispatch(events, [channel_a, channel_b])
    assert channel_a.received == events
    assert channel_b.received == events


def test_webhook_channel_records_the_payload_it_would_have_sent():
    webhook = WebhookChannel()
    event = {"day": 3, "metric": "error_rate", "detectors": ["threshold"]}
    dispatch([event], [webhook])
    assert webhook.sent == [event]
