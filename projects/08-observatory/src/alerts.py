"""
Turning detector flags into alerts, and deciding when not to. A metric
that's been flagged for ten days straight shouldn't produce ten
alerts, it should produce one, on the day it first crossed the line.
find_alert_events() collapses each metric's flagged days into events on
the rising edge only; dispatch() fans each event out to every
configured channel, so adding a new destination is a one-class
addition, same idea as the detectors.

Channels behind one interface:
- ConsoleChannel: prints to stdout.
- FileChannel: appends one JSON line per event to a log file.
- WebhookChannel: a stand-in for a real outbound webhook (Slack,
  PagerDuty, etc). No live network call here, this repo has no
  external dependencies by design, so it records the payload it would
  have sent instead.
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


def find_alert_events(flags_df: pd.DataFrame) -> list[dict]:
    """One event per metric per contiguous run of flagged days, dated
    to the day the run starts, not every day it continues. flags_df is
    long-form: day, metric, detector, flagged (as returned by
    detectors.run_detectors)."""
    events = []
    for metric, group in flags_df.groupby("metric"):
        pivot = group.pivot(index="day", columns="detector", values="flagged").fillna(False).sort_index()
        any_flagged = pivot.any(axis=1)
        was_flagged = False
        for day, flagged in any_flagged.items():
            if flagged and not was_flagged:
                triggered = sorted(pivot.columns[pivot.loc[day]].tolist())
                events.append({"day": int(day), "metric": metric, "detectors": triggered})
            was_flagged = flagged
    return sorted(events, key=lambda e: (e["day"], e["metric"]))


class AlertChannel(ABC):
    @abstractmethod
    def send(self, event: dict) -> None:
        ...


class ConsoleChannel(AlertChannel):
    def send(self, event: dict) -> None:
        detectors = ", ".join(event["detectors"])
        print(f"[ALERT] day {event['day']}: {event['metric']} flagged by {detectors}")


class FileChannel(AlertChannel):
    def __init__(self, path: Path):
        self.path = Path(path)

    def send(self, event: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")


class WebhookChannel(AlertChannel):
    def __init__(self):
        self.sent: list[dict] = []

    def send(self, event: dict) -> None:
        self.sent.append(event)


def dispatch(events: list[dict], channels: list[AlertChannel]) -> None:
    for event in events:
        for channel in channels:
            channel.send(event)
