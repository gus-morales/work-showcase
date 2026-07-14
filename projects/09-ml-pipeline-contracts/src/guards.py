"""Point-in-time and leakage guards, the "things that quietly ruin a
model" the Feature stage has to get right. Kept as standalone
functions, not buried inside the Feature stage, so they can be unit
tested directly against a case that would leak.
"""
import pandas as pd


def filter_point_in_time(events: pd.DataFrame, cutoffs: pd.Series, join_key: str, event_date_col: str) -> pd.DataFrame:
    """Drop every event row dated after its identifier's decision-time
    cutoff. `cutoffs` is a Series indexed by the identifier, one cutoff
    per identifier (e.g. training_data.set_index(identifier_col)[cutoff_col]).
    Using the unfiltered `events` table directly, instead of this, is
    exactly the mistake that lets post-decision information leak into a
    feature: a customer's usage *after* the day the model is scoring
    them, or a ticket's activity *after* the triage window closed.
    """
    aligned_cutoff = events[join_key].map(cutoffs)
    if aligned_cutoff.isna().any():
        missing = events.loc[aligned_cutoff.isna(), join_key].unique()
        raise ValueError(f"no cutoff found for {join_key} value(s): {list(missing)[:5]}")
    return events[events[event_date_col] <= aligned_cutoff].copy()


def assert_no_target_leakage(feature_cols: list[str], target_names: list[str]) -> None:
    """A feature column that's literally the target (or shares its raw
    name) has no business being a feature. This is the same guard
    `stage_io.verify_feature_stage_output` re-checks at the contract
    boundary; it also lives here so the Feature stage can catch it
    before it ever writes an artifact, not just after."""
    leaked = set(feature_cols) & set(target_names)
    if leaked:
        raise ValueError(f"feature columns {sorted(leaked)} are also target names, that's target leakage")
