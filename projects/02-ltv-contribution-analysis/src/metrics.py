"""
Semantic layer: one governed definition per business metric, so "GMV"
and "revenue" can't quietly mean different things in different SQL
files or Python scripts. BNPL take-rate economics make this an easy
mix-up: GMV is the gross value moving through the platform (what a
customer's orders are worth), revenue is the company's own cut of
that (order_value_usd times the take rate), and a customer's order
total looks like "their revenue" until the take rate is accounted
for. sql/*.sql and src/*.py are expected to source their aggregation
formulas from here rather than redefining them inline.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str
    description: str
    sql_expr: str  # canonical SQL aggregation expression
    column: str | None  # underlying source column, where there is a single one


GMV = Metric(
    name="gmv",
    description="Gross merchandise value: total order value moving through the platform, before the take rate.",
    sql_expr="SUM(order_value_usd)",
    column="order_value_usd",
)

REVENUE = Metric(
    name="revenue",
    description="The company's own take: merchant/interest fee revenue, order_value_usd times the take rate.",
    sql_expr="SUM(fee_revenue_usd)",
    column="fee_revenue_usd",
)

ACTIVE_CUSTOMERS = Metric(
    name="active_customers",
    description="Distinct customers with at least one order in the period.",
    sql_expr="COUNT(DISTINCT customer_id)",
    column="customer_id",
)

ORDERS = Metric(
    name="orders",
    description="Order count in the period.",
    sql_expr="COUNT(*)",
    column="order_id",
)

ORDERS_PER_CUSTOMER = Metric(
    name="orders_per_customer",
    description="Orders divided by active customers: the frequency driver in the GMV = customers x frequency x AOV decomposition.",
    sql_expr="COUNT(*) * 1.0 / COUNT(DISTINCT customer_id)",
    column=None,
)

AOV = Metric(
    name="avg_order_value",
    description="Average order value: GMV divided by orders, the third driver in the GMV decomposition.",
    sql_expr="SUM(order_value_usd) / COUNT(*)",
    column=None,
)

RETENTION_RATE = Metric(
    name="retention_rate",
    description="Active customers in a cohort-month, divided by that cohort's original size.",
    sql_expr="active_customers / cohort_size",
    column=None,
)

REGISTRY = {m.name: m for m in [GMV, REVENUE, ACTIVE_CUSTOMERS, ORDERS, ORDERS_PER_CUSTOMER, AOV, RETENTION_RATE]}


def get_metric(name: str) -> Metric:
    if name not in REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Governed metrics: {sorted(REGISTRY)}")
    return REGISTRY[name]


def glossary() -> str:
    return "\n".join(f"{m.name}: {m.description}" for m in REGISTRY.values())


_SUM_COLUMN_PATTERN = re.compile(r"SUM\(\s*(?:\w+\.)?(\w+)\s*\)")
_REVENUE_ALIAS_PATTERN = re.compile(r"AS\s+(\w*revenue\w*)", re.IGNORECASE)
_GMV_ALIAS_PATTERN = re.compile(r"AS\s+(\w*gmv\w*)", re.IGNORECASE)


def check_sql_uses_governed_metrics(sql_text: str) -> list[str]:
    """Scans a .sql file line by line for any column aliased like
    "revenue" or "gmv" and checks that its SUM() argument matches the
    governed definition above, catching exactly the kind of mix-up
    where a revenue-labeled column is quietly aggregating GMV (or vice
    versa). A per-line heuristic, not a full SQL parser: it assumes
    one aggregated expression per line, which is this project's actual
    SQL style throughout sql/*.sql. A revenue-labeled column
    re-aggregating an already revenue-labeled subtotal from an inner
    CTE (or the reverse for
    GMV) isn't flagged: that's summing a value whose lineage was
    already checked, not a raw-column mix-up. Returns a list of
    violation messages, empty if consistent."""
    violations = []
    for line in sql_text.splitlines():
        sum_match = _SUM_COLUMN_PATTERN.search(line)
        if not sum_match:
            continue
        col = sum_match.group(1)

        revenue_alias = _REVENUE_ALIAS_PATTERN.search(line)
        if revenue_alias and col == GMV.column:
            violations.append(
                f"'{revenue_alias.group(1)}' aggregates '{col}', expected '{REVENUE.column}' "
                "(the governed revenue metric)"
            )

        gmv_alias = _GMV_ALIAS_PATTERN.search(line)
        if gmv_alias and col == REVENUE.column:
            violations.append(
                f"'{gmv_alias.group(1)}' aggregates '{col}', expected '{GMV.column}' "
                "(the governed GMV metric)"
            )
    return violations
