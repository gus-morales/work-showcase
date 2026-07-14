"""
The metric catalog: what's monitored, and how. Every ops metric or
model feature the pipeline watches gets one YAML file under `catalog/`
instead of a hardcoded constant, so adding or reconfiguring a monitored
signal is a one-file change, not a code change. `snapshot.py` builds
both detection engines' configuration directly from this catalog: an
ops metric's threshold limit, a model feature's dtype hint for popmon,
all of it comes from here.
"""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

Kind = Literal["ops_metric", "model_feature"]
Dtype = Literal["numerical", "categorical"]
Direction = Literal["above", "below"]


class CatalogEntry(BaseModel):
    name: str
    kind: Kind
    description: str
    monitor: bool = True
    dtype: Dtype | None = None
    threshold_limit: float | None = None
    threshold_direction: Direction | None = None

    @model_validator(mode="after")
    def _check_consistency(self):
        if self.kind == "ops_metric":
            if self.threshold_limit is None or self.threshold_direction is None:
                raise ValueError("kind='ops_metric' requires threshold_limit and threshold_direction")
            if self.dtype is not None:
                raise ValueError("kind='ops_metric' should not set dtype")
        else:  # model_feature
            if self.dtype is None:
                raise ValueError("kind='model_feature' requires dtype")
            if self.threshold_limit is not None or self.threshold_direction is not None:
                raise ValueError("kind='model_feature' should not set threshold_limit/threshold_direction")
        return self


def load_catalog(catalog_dir: Path) -> list[CatalogEntry]:
    """Parse every *.yaml file under catalog_dir into a CatalogEntry."""
    entries = []
    for path in sorted(catalog_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        entries.append(CatalogEntry(**raw))
    return entries


def ops_metrics(entries: list[CatalogEntry]) -> list[CatalogEntry]:
    return [e for e in entries if e.kind == "ops_metric" and e.monitor]


def model_features(entries: list[CatalogEntry]) -> list[CatalogEntry]:
    return [e for e in entries if e.kind == "model_feature" and e.monitor]
