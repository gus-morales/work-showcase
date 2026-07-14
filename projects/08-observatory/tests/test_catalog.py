"""Unit tests for the metric catalog: the record contract for ops
metrics vs. model features, and the real catalog/ directory."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from catalog import CatalogEntry, load_catalog, model_features, ops_metrics

BASE = Path(__file__).resolve().parents[1]


def test_valid_ops_metric_passes():
    CatalogEntry(name="x", kind="ops_metric", description="d", threshold_limit=1.0, threshold_direction="above")


def test_valid_model_feature_passes():
    CatalogEntry(name="x", kind="model_feature", description="d", dtype="numerical")


def test_ops_metric_requires_threshold():
    with pytest.raises(ValidationError):
        CatalogEntry(name="x", kind="ops_metric", description="d")


def test_ops_metric_cannot_set_dtype():
    with pytest.raises(ValidationError):
        CatalogEntry(name="x", kind="ops_metric", description="d", threshold_limit=1.0,
                     threshold_direction="above", dtype="numerical")


def test_model_feature_requires_dtype():
    with pytest.raises(ValidationError):
        CatalogEntry(name="x", kind="model_feature", description="d")


def test_model_feature_cannot_set_threshold():
    with pytest.raises(ValidationError):
        CatalogEntry(name="x", kind="model_feature", description="d", dtype="numerical",
                     threshold_limit=1.0, threshold_direction="above")


def test_real_catalog_loads_and_splits_correctly():
    entries = load_catalog(BASE / "catalog")
    assert len(ops_metrics(entries)) == 4
    assert len(model_features(entries)) == 5


def test_unmonitored_entries_are_excluded():
    entries = [
        CatalogEntry(name="a", kind="ops_metric", description="d", threshold_limit=1.0,
                     threshold_direction="above", monitor=False),
        CatalogEntry(name="b", kind="ops_metric", description="d", threshold_limit=1.0,
                     threshold_direction="above", monitor=True),
    ]
    assert [e.name for e in ops_metrics(entries)] == ["b"]
