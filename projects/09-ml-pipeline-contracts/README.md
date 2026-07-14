# ML Pipeline Contracts

Building a machine learning model usually happens in stages: understand the raw data,
assemble a clean training set, build the actual inputs (the "features") a model will
learn from, train the model, then check whether it's good enough to use. On a real
team, different people often own different stages. That handoff is where things quietly
go wrong. A common, easy-to-miss mistake: whoever builds the features doesn't realize
that a column they've included is actually the answer the model is supposed to
predict, not a legitimate signal for it. The model then looks excellent in testing and
fails the moment it meets real data, because it was never predicting anything, it was
looking up the answer.

This project builds that five-stage pipeline with a check at every handoff: before one
stage's output is allowed to feed the next stage, code opens the actual file that got
produced and checks it against a stated rule, not just that the previous step ran
without crashing. When a handoff breaks a rule, for example a stage that hands over the
target itself disguised as a feature, the pipeline stops immediately and says exactly
what went wrong and where, instead of quietly training a broken model.

> All data here is synthetically generated. No proprietary data, methodology, or
> results from any employer are used or implied. Every stage runs as a plain,
> deterministic Python function rather than a live model call, so re-running the
> pipeline produces the same result every time.

**Skills and tools featured:**

- Contracts between pipeline stages (Pydantic) that open each stage's actual output file and check it, not just that the code ran without error
- A single sign-off document that has to be explicitly finalized before any stage is allowed to run
- A filter that strips out any data a decision-maker couldn't have known about yet, tested against a case built to leak
- The exact same pipeline run against two unrelated synthetic business problems, one with a single yes/no target, one with a yes/no target and a numeric target together
- Information Value and Population Stability Index (PSI), two standard scoring-model diagnostics, computed automatically for every feature a model ends up using
- A deliberately broken run, with the real error message it produces, showing the checks actually catch a mistake rather than just claiming to

## The problem

A five-stage pipeline where nothing checks the boundaries between stages tends to
break in the same boring way: one stage assumes a column from the previous stage is
named a certain thing, or assumes a certain column is a safe input and not the answer
itself, and nothing catches that assumption until training either crashes outright or,
worse, runs fine and trains on the answer. Adding more code inside each stage doesn't
fix this, since no single stage can see what the others actually produced. What fixes
it is a check placed at the boundary: a stated promise of what a stage's output should
contain, and code that opens the file and confirms the promise holds before the
pipeline is allowed to continue.

## How it works

Every run starts with one document, `MODEL_SCOPE.md`, that states what's being built:
what it's for, what's being predicted (the "target"), how success will be measured,
and a cap on how many features the model is allowed to use. That document has a
status, `draft`, `signed_off`, or `frozen`, and nothing else in the pipeline is allowed
to run until it's `frozen`, the same way a plan gets finalized and signed off before
anyone starts building against it.

![One plan document, five checked handoffs](assets/architecture.png)

The plan document states intent, not the mechanics of where the data actually lives.
Two more files sit next to it: `bindings.yaml` (which real files back the data the
plan refers to, and how they connect to each other) and `feature_spec.yaml` (the
specific features someone wants built, e.g. "average number of logins before the
decision was made"). One script, `src/orchestrator.py`, runs the five stages in order.
Between every stage, it opens the file the stage just produced and checks specific
things about it before letting the next stage start:

| Stage | What it does | What gets checked before moving on |
|---|---|---|
| ① Understand the data | Profiles the raw data and reports how the target behaves over time | The report files were actually written |
| ② Prepare the data | Builds one clean table: one row per customer or ticket, plus the target, nothing else | The identifier column is real with no duplicates, the target has no missing values, and the row count matches what was reported |
| ③ Build features | Builds the actual model inputs, dropping anything dated after the decision point | The features line up with the same identifier as the previous stage, the target hasn't snuck in as a feature, and the feature count is under the agreed cap |
| ④ Train | Fits a model per target, then repeatedly drops the weakest feature to see if a smaller model does just as well | Every feature the model claims to use actually came from the previous stage, and the reported score is a real number |
| ⑤ Check against the bar | Compares the trained model's score to the number stated in the plan document | The pass/fail result is actually consistent with the score and the bar it's being measured against |

The data-preparation stage only builds the identifier-and-target table, on purpose. A
stage that both builds features and decides what the target is has no natural place
for a leakage check to sit, since it's the one deciding what counts as leakage. Feature
building, and only feature building, is allowed to touch the detailed event-level data.

## Not training on hindsight

The two example problems below both have event-level data, individual login events or
individual ticket updates, that include records dated both before and after the moment
a real decision would have needed to be made. Before those records get turned into
features, anything dated after the decision point gets dropped. Skipping that step is
an easy mistake with a specific consequence: the model gets to see information nobody
would have actually had at decision time, so it looks accurate in testing and then
fails once it's making real decisions on data that hasn't happened yet. A test in this
project shows the difference directly: computed the ordinary way, one future event
pulls a customer's average logins from 1 up to 50; computed the correct way, filtering
out anything after the decision date, it stays at 1.

## Two example problems, run through the same pipeline unchanged

**Predicting subscription cancellations** (`runs/churn/run-001`): a subscription
business trying to flag, around each customer's 90-day mark, who's likely to cancel in
the next 30 days. 3,000 customers, 29% actually cancel. A model built from 10
candidate signals, narrowed down to the strongest 6, correctly ranks cancelling
customers as risky well above chance: a score of 0.39 out of 1 (PR-AUC, a measure of
how well the model ranks people who actually cancel above those who don't; guessing at
random would score around the 29% cancellation rate), clearing the 0.30 bar the plan
called for.

**Predicting how a support ticket will go** (`runs/ticket-triage/run-001`): a support
team trying to decide, within 2 hours of a ticket opening, whether it's headed for
escalation and roughly how long it will take to close. 2,500 tickets, 35% eventually
escalate, the typical ticket takes 10.5 hours to close. This plan asks for two
different kinds of prediction from the same data at once: a yes/no call on escalation
(scores 0.44 vs. a required 0.40) and a number, hours to close (off by 3.05 hours on
average vs. a 4.5-hour requirement). None of the pipeline code changed to support
asking for two predictions instead of one, the plan document itself enforces that a
request for both a yes/no answer and a number gets exactly one of each, with a bar
stated for both.

Every number above came from actually running the pipeline against the example folders
committed in this repo, not rounded or invented for effect.
`notebooks/09_ml_pipeline_contracts.ipynb` reads the same files and charts them.

## What the training stage reports

Alongside the trained model, the training stage writes out, for every feature it kept:
how much that feature actually contributed to the model's predictions, and two extra
numbers borrowed from credit-scoring practice. Information Value measures how cleanly
a feature, on its own, separates the two outcomes of a yes/no target. PSI (Population
Stability Index) checks whether a feature looks the same in the data used to train the
model as in the data held back to test it, a big difference there would mean the two
halves of the data don't actually resemble each other, which would make the whole
evaluation untrustworthy.

## Proving the safety check actually works

It's easy to claim a safety check works without showing it. `src/demo_broken_contract.py`
runs the first two stages for real, then deliberately hands the training stage a
feature file where the target itself, the exact thing being predicted, is included as
one of the "features." This is a common real mistake: someone builds a join, forgets
to drop the answer column, and the resulting model looks perfect because it's just
looking up the label. Real, unedited output from running that script:

```
ContractViolation: [features] target column(s) ['churned_next_30d'] appear in feature_cols, that's target leakage
```

The pipeline stops right there. No model gets trained on the broken file, and a test
in this project checks that directly: after this exact failure, no model file exists
on disk anywhere.

## Starting a new run

```
$ python src/new_run.py --slug expansion-risk --run-id run-001 \
    --name "Expansion Risk" --objective "..." --problem "..." \
    --task-type classification --max-features 10
Wrote runs/expansion-risk/run-001/MODEL_SCOPE.md (status: draft)
```

This writes a starting plan document with status `draft`. Someone fills in the actual
target, the success bar, and the two execution files by hand, then sets the status to
`frozen` once it's been reviewed, that's the step that actually unlocks the pipeline
for that run.

## Repo layout

- `src/model_scope.py`: reads and validates the plan document (`MODEL_SCOPE.md`).
- `src/stage_io.py`: what each stage promises to produce, and the checks that confirm it did.
- `src/bindings.py`: loaders for a run's `bindings.yaml` and `feature_spec.yaml`.
- `src/guards.py`: the point-in-time filter and the target-leakage check, both standalone and tested on their own.
- `src/generate_data.py`: synthetic source data for both example problems.
- `src/stages/`: the five stage implementations (`eda.py`, `data_prep.py`, `feature_engineering.py`, `training.py`, `validation.py`).
- `src/orchestrator.py`: runs the five stages in order, checking every handoff; `python src/orchestrator.py --run-dir <path>`.
- `src/new_run.py`: creates a new run's starting plan document.
- `src/demo_broken_contract.py`: the deliberately broken handoff, with real captured output.
- `src/render_architecture.py`: renders `assets/architecture.png`.
- `runs/<slug>/<run-id>/`: the plan document, the two execution files, and the output of every stage. `runs/churn/` and `runs/ticket-triage/` are the two example problems; `runs/broken-example/` is the caught-mistake example.
- `notebooks/09_ml_pipeline_contracts.ipynb`: reads the committed run output and charts it; it doesn't call any stage or re-run anything itself.
- `tests/`: covers the plan-document rules, every handoff check (including cases built to fail), the point-in-time filter, the scoring-diagnostic math, and two full pipeline runs, one clean, one deliberately broken partway through.

## Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/orchestrator.py --run-dir runs/churn/run-001
python src/orchestrator.py --run-dir runs/ticket-triage/run-001
python src/demo_broken_contract.py
jupyter nbconvert --to notebook --execute --inplace notebooks/09_ml_pipeline_contracts.ipynb
```

`data/` is regenerated by the first command rather than committed, since it's large and
easy to reproduce. Under `runs/`, the trained model files and the large intermediate
tables are regenerated the same way; everything else, the plan documents, the small
reports, the feature-importance tables, is kept in version control.

## Tests

```bash
pytest tests/ -v
```

Runs in CI on every push (see the badge at the [repo root](../../README.md)).
