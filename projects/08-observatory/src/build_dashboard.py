"""Builds dashboard/index.html, a static, self-contained snapshot of the
monitoring run: KPI cards, an alerts-over-time chart, a detector-mix
chart, and a per-metric status table. Like everything else downstream
of the pipeline, this only reads snapshot/flags.csv, snapshot/events.json,
and catalog/*.yaml, it never touches the database or reruns a detector.
This is a point-in-time build, not a live app: rerun it after
snapshot.py to refresh the numbers.

Run:
    python src/build_dashboard.py
"""
import json
from pathlib import Path

import pandas as pd

from catalog import load_catalog

BASE = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = BASE / "snapshot"
CATALOG_DIR = BASE / "catalog"
DASHBOARD_DIR = BASE / "dashboard"

POPMON = "popmon"
OPS_DETECTORS = {"threshold", "zscore", "trend_break", "data_gap"}


def _engine(detector: str) -> str:
    return "model" if detector == POPMON else "ops"


def build_metric_rows(catalog_entries, flags_df: pd.DataFrame) -> list[dict]:
    rows = []
    for entry in catalog_entries:
        metric_flags = flags_df[flags_df["metric"] == entry.name]
        flagged = metric_flags[metric_flags["flagged"]]
        triggered = sorted(flagged["detector"].unique().tolist())
        first_day = int(flagged["day"].min()) if len(flagged) else None
        rows.append({
            "name": entry.name,
            "kind": entry.kind,
            "description": entry.description,
            "engine": "popmon" if entry.kind == "model_feature" else "ops detectors",
            "status": "flagged" if triggered else "clean",
            "detectors": triggered,
            "first_flagged_day": first_day,
        })
    return sorted(rows, key=lambda r: (r["first_flagged_day"] is None, r["first_flagged_day"] or 0))


def build_daily_alert_counts(flags_df: pd.DataFrame) -> list[dict]:
    """Distinct metrics with at least one flagged detector that day, split
    by which engine caught it, one row per day covered by flags_df."""
    flagged = flags_df[flags_df["flagged"]].copy()
    flagged["engine"] = flagged["detector"].map(_engine)
    per_day = (
        flagged.drop_duplicates(["day", "metric", "engine"])
        .groupby(["day", "engine"])
        .size()
        .unstack(fill_value=0)
    )
    all_days = pd.RangeIndex(flags_df["day"].min(), flags_df["day"].max() + 1)
    per_day = per_day.reindex(all_days, fill_value=0)
    return [
        {"day": int(day), "ops": int(row.get("ops", 0)), "model": int(row.get("model", 0))}
        for day, row in per_day.iterrows()
    ]


def build_detector_breakdown(events: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for event in events:
        for detector in event["detectors"]:
            counts[detector] = counts.get(detector, 0) + 1
    return [{"detector": name, "count": count} for name, count in sorted(counts.items())]


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Observatory — monitoring snapshot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@500&family=Lato:wght@400;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1" integrity="sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ" crossorigin="anonymous"></script>
<style>
  :root {
    --ink: #ECECEA; --slate: #8FBBDB; --teal: #82C2B7; --amber: #D8AD72;
    --red: #C97B6E; --grey: #8B8B87; --light-grey: #2E3136;
    --bg: #17191C; --card-bg: #1E2124; --chip-bg: #262B30;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: 'Lato', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 48px 28px 56px; }
  header { margin-bottom: 32px; }
  h1 { font-family: 'Lora', serif; font-weight: 500; font-size: 30px; margin: 0 0 8px; }
  .subtitle { color: #A5A5A1; font-size: 15px; max-width: 720px; line-height: 1.5; }
  .badge {
    display: inline-block; margin-top: 14px; padding: 4px 11px; border-radius: 20px;
    background: var(--chip-bg); border: 1px solid var(--light-grey); color: var(--grey);
    font-size: 12px; letter-spacing: 0.3px;
  }
  .kpi-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px; margin: 28px 0;
  }
  .kpi-card {
    background: var(--card-bg); border: 1px solid var(--light-grey); border-radius: 10px;
    padding: 16px 18px;
  }
  .kpi-label { font-size: 11.5px; color: var(--grey); text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi-value { font-family: 'Lora', serif; font-size: 26px; margin-top: 6px; color: var(--ink); }
  .kpi-note { font-size: 11.5px; color: var(--grey); margin-top: 4px; }
  .chart-row { display: grid; grid-template-columns: 1.6fr 1fr; gap: 16px; margin-bottom: 16px; }
  .panel {
    background: var(--card-bg); border: 1px solid var(--light-grey); border-radius: 10px;
    padding: 20px 22px;
  }
  .panel h2 { font-family: 'Lora', serif; font-weight: 500; font-size: 17px; margin: 0 0 3px; }
  .panel .panel-sub { font-size: 12.5px; color: var(--grey); margin-bottom: 14px; }
  canvas { max-height: 280px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th {
    text-align: left; padding: 8px 10px; color: var(--grey); font-weight: 600;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px;
    border-bottom: 1px solid var(--light-grey);
  }
  tbody td { padding: 9px 10px; border-bottom: 1px solid var(--light-grey); vertical-align: top; }
  tbody tr:last-child td { border-bottom: none; }
  .status-pill {
    display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 11.5px;
    letter-spacing: 0.3px;
  }
  .status-flagged { background: rgba(201,123,110,0.18); color: var(--red); }
  .status-clean { background: rgba(130,194,183,0.18); color: var(--teal); }
  .det-chip {
    display: inline-block; background: var(--chip-bg); border: 1px solid var(--light-grey);
    border-radius: 5px; padding: 1px 7px; margin: 1px 3px 1px 0; font-size: 11px; color: #C9C9C4;
  }
  .metric-desc { color: var(--grey); font-size: 12px; }
  footer { margin-top: 32px; font-size: 12.5px; color: var(--grey); line-height: 1.7; }
  footer a { color: var(--slate); text-decoration: none; }
  footer a:hover { text-decoration: underline; }
  code { background: var(--chip-bg); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
  @media (max-width: 820px) { .chart-row { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Observatory</h1>
    <div class="subtitle">
      A monitoring snapshot for a DS team's operations: pipeline/ops metrics tracked by
      four scalar detectors, model-feature and prediction drift tracked by
      <a href="https://github.com/ing-bank/popmon" style="color:var(--slate);text-decoration:none;">popmon</a>,
      both configured from one metric catalog and unified into one alert stream.
    </div>
    <span class="badge">Static build from a single snapshot run — not a live app. See README for how to regenerate.</span>
  </header>

  <section class="kpi-row" id="kpi-row"></section>

  <section class="chart-row">
    <div class="panel">
      <h2>Alerting metrics per day</h2>
      <div class="panel-sub">Distinct metrics with at least one flagged detector that day, split by engine</div>
      <canvas id="alerts-chart"></canvas>
    </div>
    <div class="panel">
      <h2>Alert events by detector</h2>
      <div class="panel-sub">Which check first caught each event</div>
      <canvas id="detector-chart"></canvas>
    </div>
  </section>

  <section class="panel" style="margin-bottom:16px;">
    <h2>Metric status</h2>
    <div class="panel-sub">Every monitored signal in the catalog, ops metrics and model features together</div>
    <table id="metric-table">
      <thead><tr>
        <th>Metric</th><th>Engine</th><th>Status</th><th>First alert</th><th>Detector(s)</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Alert timeline</h2>
    <div class="panel-sub">Every alert event, in order, one row per rising edge (not per flagged day)</div>
    <table id="events-table">
      <thead><tr><th>Day</th><th>Metric</th><th>Detector(s)</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <footer>
    Full per-feature popmon detail: <a href="../snapshot/popmon_stability_report.html">popmon_stability_report.html ↗</a><br>
    Regenerate this snapshot: <code>python src/generate_data.py && python src/snapshot.py && python src/build_dashboard.py</code>
  </footer>
</div>

<script>
const KPIS = __KPIS__;
const DAILY = __DAILY__;
const DETECTOR_BREAKDOWN = __DETECTOR_BREAKDOWN__;
const METRIC_ROWS = __METRIC_ROWS__;
const EVENTS = __EVENTS__;

const COLORS = { ops: '#8FBBDB', model: '#D8AD72', grid: '#2E3136', ink: '#ECECEA', grey: '#8B8B87' };

function renderKPIs() {
  const row = document.getElementById('kpi-row');
  row.innerHTML = KPIS.map(k => `
    <div class="kpi-card">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-note">${k.note || ''}</div>
    </div>`).join('');
}

function renderAlertsChart() {
  const ctx = document.getElementById('alerts-chart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: DAILY.map(d => d.day),
      datasets: [
        { label: 'Ops detectors', data: DAILY.map(d => d.ops), backgroundColor: COLORS.ops, stack: 's' },
        { label: 'popmon', data: DAILY.map(d => d.model), backgroundColor: COLORS.model, stack: 's' },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { color: COLORS.ink, usePointStyle: true } } },
      scales: {
        x: { stacked: true, ticks: { color: COLORS.grey, maxTicksLimit: 12 }, grid: { display: false }, title: { display: true, text: 'day', color: COLORS.grey } },
        y: { stacked: true, beginAtZero: true, ticks: { color: COLORS.grey, precision: 0 }, grid: { color: COLORS.grid } },
      }
    }
  });
}

function renderDetectorChart() {
  const ctx = document.getElementById('detector-chart').getContext('2d');
  const ops = DETECTOR_BREAKDOWN.filter(d => d.detector !== 'popmon');
  const model = DETECTOR_BREAKDOWN.filter(d => d.detector === 'popmon');
  const ordered = [...ops, ...model];
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ordered.map(d => d.detector),
      datasets: [{
        data: ordered.map(d => d.count),
        backgroundColor: ordered.map(d => d.detector === 'popmon' ? COLORS.model : COLORS.ops),
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { color: COLORS.grey, precision: 0 }, grid: { color: COLORS.grid } },
        y: { ticks: { color: COLORS.ink }, grid: { display: false } },
      }
    }
  });
}

function renderMetricTable() {
  const tbody = document.querySelector('#metric-table tbody');
  tbody.innerHTML = METRIC_ROWS.map(r => `
    <tr>
      <td><strong>${r.name}</strong><div class="metric-desc">${r.description}</div></td>
      <td>${r.engine}</td>
      <td><span class="status-pill status-${r.status}">${r.status}</span></td>
      <td>${r.first_flagged_day === null ? '—' : 'day ' + r.first_flagged_day}</td>
      <td>${r.detectors.map(d => `<span class="det-chip">${d}</span>`).join('') || '—'}</td>
    </tr>`).join('');
}

function renderEventsTable() {
  const tbody = document.querySelector('#events-table tbody');
  tbody.innerHTML = EVENTS.map(e => `
    <tr>
      <td>day ${e.day}</td>
      <td>${e.metric}</td>
      <td>${e.detectors.map(d => `<span class="det-chip">${d}</span>`).join('')}</td>
    </tr>`).join('');
}

renderKPIs();
renderAlertsChart();
renderDetectorChart();
renderMetricTable();
renderEventsTable();
</script>
</body>
</html>
"""


def main():
    catalog_entries = load_catalog(CATALOG_DIR)
    flags_df = pd.read_csv(SNAPSHOT_DIR / "flags.csv")
    events = json.loads((SNAPSHOT_DIR / "events.json").read_text())

    ops_count = sum(1 for e in catalog_entries if e.kind == "ops_metric" and e.monitor)
    model_count = sum(1 for e in catalog_entries if e.kind == "model_feature" and e.monitor)
    n_days = int(flags_df["day"].max()) + 1
    flagged_metrics = {e["metric"] for e in events}

    kpis = [
        {"label": "Metrics monitored", "value": ops_count + model_count,
         "note": f"{ops_count} ops · {model_count} model"},
        {"label": "Days observed", "value": n_days},
        {"label": "Alert events", "value": len(events)},
        {"label": "Metrics ever flagged", "value": f"{len(flagged_metrics)} / {ops_count + model_count}"},
        {"label": "Detection engines", "value": 2, "note": "detectors + popmon"},
    ]

    metric_rows = build_metric_rows(catalog_entries, flags_df)
    daily = build_daily_alert_counts(flags_df)
    detector_breakdown = build_detector_breakdown(events)

    html = (
        TEMPLATE
        .replace("__KPIS__", json.dumps(kpis))
        .replace("__DAILY__", json.dumps(daily))
        .replace("__DETECTOR_BREAKDOWN__", json.dumps(detector_breakdown))
        .replace("__METRIC_ROWS__", json.dumps(metric_rows))
        .replace("__EVENTS__", json.dumps(events))
    )

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / "index.html"
    out_path.write_text(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
