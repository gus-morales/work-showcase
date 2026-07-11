"""Unit tests for the semantic layer: the metric registry itself, and
the SQL-consistency check against both hand-built SQL snippets and the
project's actual sql/*.sql files (so real drift gets caught, not just
the hypothetical kind)."""
from pathlib import Path

import pytest

from metrics import GMV, REVENUE, get_metric, glossary, check_sql_uses_governed_metrics

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def test_gmv_and_revenue_are_different_columns():
    # The whole point of the registry: these must never collapse to
    # the same underlying column.
    assert GMV.column != REVENUE.column
    assert GMV.sql_expr != REVENUE.sql_expr


def test_get_metric_returns_known_metric():
    assert get_metric("gmv") is GMV
    assert get_metric("revenue") is REVENUE


def test_get_metric_raises_on_unknown_name():
    with pytest.raises(KeyError):
        get_metric("not_a_real_metric")


def test_glossary_mentions_every_registered_metric():
    text = glossary()
    assert "gmv" in text
    assert "revenue" in text
    assert "active_customers" in text


def test_flags_revenue_alias_aggregating_gmv_column():
    sql = "SELECT SUM(order_value_usd) AS total_revenue_usd FROM orders;"
    violations = check_sql_uses_governed_metrics(sql)
    assert len(violations) == 1
    assert "order_value_usd" in violations[0]


def test_flags_gmv_alias_aggregating_revenue_column():
    sql = "SELECT SUM(fee_revenue_usd) AS total_gmv_usd FROM orders;"
    violations = check_sql_uses_governed_metrics(sql)
    assert len(violations) == 1
    assert "fee_revenue_usd" in violations[0]


def test_passes_correctly_aggregated_metrics():
    # One aggregated expression per line, matching this project's SQL
    # style; the checker is a per-line heuristic, not a full parser.
    sql = """
    SELECT
        SUM(order_value_usd) AS gmv_usd,
        SUM(fee_revenue_usd) AS revenue_usd
    FROM orders;
    """
    assert check_sql_uses_governed_metrics(sql) == []


def test_does_not_flag_reaggregating_an_already_named_subtotal():
    # An outer query summing an inner CTE's already-correct
    # revenue_usd column is still "revenue", not a raw-column mix-up.
    sql = """
    WITH per_customer AS (SELECT customer_id, SUM(fee_revenue_usd) AS revenue_usd FROM orders GROUP BY 1)
    SELECT SUM(revenue_usd) AS total_revenue_usd FROM per_customer;
    """
    assert check_sql_uses_governed_metrics(sql) == []


@pytest.mark.parametrize("filename", [
    "01_cohort_revenue.sql", "02_monthly_kpis.sql", "03_channel_quality.sql", "04_channel_mix_shift.sql",
])
def test_project_sql_files_use_governed_metrics(filename):
    sql_text = (SQL_DIR / filename).read_text()
    assert check_sql_uses_governed_metrics(sql_text) == []
