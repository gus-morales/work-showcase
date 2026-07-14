# Data Science Decision Governance

A working record format for a data science team's own decisions: model launches, experiment rollouts, dashboard changes, pipeline changes, metric-definition changes, deprecations. One decision is one file. Impact level (low/medium/high) sets how much rigor that file needs, enforced by a schema instead of by memory. This README documents the standard; `decisions/` holds real example records written against it.

> Everything here, the example records included, is fictional. No proprietary data, decisions, or results from any employer are used or implied.

**Skills and tools featured:**

- Schema-based record validation (Pydantic)
- CLI tooling to scaffold and validate records
- A live scan for monitoring commitments that are currently overdue

## The problem

A data science team makes a lot of decisions that never get written down anywhere: launching a model, rolling out an experiment, changing a dashboard, changing a pipeline, redefining a metric, deprecating something old. Without a record, two things tend to break. Nobody can later reconstruct what shipped, when, or why. And the follow-up, checking that a change actually did what it was supposed to, quietly gets skipped, especially for changes that felt too small to worry about at the time.

## How it works

Every decision is **low, medium, or high** impact. That's the only thing that determines how much process it needs.

| | low | medium | high |
|---|---|---|---|
| Ship check (did it ship as intended) | - | required | required |
| Outcome check (did it work) | required | required | required |
| Rollback plan (body section) | - | if not easily reversible | required |
| Reviewers | 1 | 2 | 3, incl. a review-board reviewer |

A decision moves through a fixed lifecycle. `abandoned` and `reverted` are the two ways out that aren't "it worked":

![Decision lifecycle](assets/lifecycle.png)

## The record

A decision is a markdown file: YAML frontmatter (the fields the schema checks) plus a free-text body (what changed, why, the rollback plan, monitoring notes). Templates for each impact level live in `templates/`. `routing.yaml` holds the reviewer-count minimums the table above is drawn from.

## The record contract

`schema.py` defines what a valid record looks like, and `validate.py` checks every file in `decisions/` against it: a low-impact record claiming a ship_check, a reverted record with no rollback outcome, a high-impact record with no rollback plan section, an approved record without enough reviewers, all fail. This is the same job a CI check would do on a pull request in a real repo; here it's a standalone script instead of a merge gate.

```
$ python src/validate.py
8 / 8 records valid.
```

## Creating a new decision

```
$ python src/new_decision.py --domain marketing --impact-level medium \
    --title "Switch the campaign dashboard to weekly cohorts" --author "J. Okafor"
Wrote decisions/marketing/DSG-0009-switch-the-campaign-dashboard-to-weekly-cohorts.md
```

Copies the right template, assigns the next id, and fills in what it can. The rest (dates once approved and shipped, reviewers, the body) gets filled in by hand as the decision moves through its lifecycle.

## Catching what's overdue right now

`open_loops.py` walks every record and reports which monitoring checks are past their due date and not marked done, a live status check against whatever's in `decisions/` right now, not a report over history. As of July 14, 2026:

```
$ python src/open_loops.py
4 overdue check(s):

DSG-0008 (decisions/product_analytics/DSG-0008-churn-model-decommission.md): ship_check was due 2026-04-17, 88 day(s) ago - "Decommission the 2023 churn-risk model"
DSG-0008 (decisions/product_analytics/DSG-0008-churn-model-decommission.md): outcome_check was due 2026-05-10, 65 day(s) ago - "Decommission the 2023 churn-risk model"
DSG-0004 (decisions/marketing/DSG-0004-campaign-attribution-pipeline-change.md): ship_check was due 2026-05-17, 58 day(s) ago - "Rebuild the campaign-attribution ETL on the new event schema"
DSG-0004 (decisions/marketing/DSG-0004-campaign-attribution-pipeline-change.md): outcome_check was due 2026-06-09, 35 day(s) ago - "Rebuild the campaign-attribution ETL on the new event schema"
```

Both flagged decisions are seeded that way on purpose, so the script has something real to catch. Re-running it later will report different numbers as `date.today()` moves and the example records don't.

## Repo layout

- `README.md`: this file, the standard.
- `templates/`: `decision-low.md`, `decision-medium.md`, `decision-high.md`.
- `routing.yaml`: reviewer-count minimums by impact level.
- `decisions/<domain>/`: 8 example records spanning all six domains, all three impact levels, and most lifecycle states (draft, shipped, closed, reverted, abandoned).
- `src/`: `schema.py` (the contract), `validate.py`, `open_loops.py`, `new_decision.py`, plus `render_lifecycle_diagram.py` for the diagram above.
- `tests/`: pytest suite covering the schema contract, the overdue-detection logic, and the scaffolding CLI.

## Reproduce

```bash
pip install -r requirements.txt
python src/validate.py
python src/open_loops.py
python src/render_lifecycle_diagram.py
```

## Tests

```bash
pytest tests/ -v
```

Runs in CI on every push (see the badge at the [repo root](../../README.md)).
