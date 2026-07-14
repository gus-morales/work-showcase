"""Loads a run's execution-detail files: `bindings.yaml` (what the
Model Scope's data-source slugs actually resolve to) and
`feature_spec.yaml` (the DS-authored feature ideas the Feature stage
compiles). These live next to the Model Scope in a run directory, not
inside it, the same intent-vs-execution split the real framework this
project is modeled on draws between a Model Scope and its bindings.
"""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class EventSource(BaseModel):
    path: str
    join_key: str
    event_date_col: str


class BasePopulation(BaseModel):
    path: str


class Bindings(BaseModel):
    identifier_col: str
    decision_time_col: str
    base_population: BasePopulation
    event_sources: dict[str, EventSource]


class FeatureIdea(BaseModel):
    name: str
    source: str          # an event_sources key, or "base"
    column: str
    agg: Literal["mean", "sum", "count", "passthrough"]
    concept: str


class FeatureSpec(BaseModel):
    features: list[FeatureIdea]


def load_bindings(path: Path) -> Bindings:
    return Bindings(**yaml.safe_load(path.read_text()))


def load_feature_spec(path: Path) -> FeatureSpec:
    return FeatureSpec(**yaml.safe_load(path.read_text()))
