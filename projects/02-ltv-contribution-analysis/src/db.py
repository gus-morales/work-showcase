"""Thin DuckDB helper: loads the generated CSVs as tables and runs the
.sql files in sql/ against them, returning pandas DataFrames."""
from pathlib import Path

import duckdb
import pandas as pd

from contracts import run_all_contracts

BASE = Path(__file__).resolve().parents[1]
SQL_DIR = BASE / "sql"
DATA_DIR = BASE / "data"


def get_connection(data_dir: Path = DATA_DIR, validate: bool = True):
    """data_dir defaults to the project's data/ folder; tests pass a
    temp directory instead so they don't depend on (or overwrite) it.
    Runs the data contracts on the CSVs before loading them, so a
    stale or hand-edited file fails loudly here instead of producing a
    quietly wrong number three scripts downstream. Set validate=False
    only for tests that are deliberately exercising a broken file."""
    if validate:
        customers = pd.read_csv(data_dir / "customers.csv")
        orders = pd.read_csv(data_dir / "orders.csv")
        run_all_contracts(customers, orders)

    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE customers AS SELECT * FROM read_csv_auto('{data_dir / 'customers.csv'}')")
    con.execute(f"CREATE TABLE orders AS SELECT * FROM read_csv_auto('{data_dir / 'orders.csv'}')")
    return con


def run_sql_file(con, filename: str) -> pd.DataFrame:
    sql = (SQL_DIR / filename).read_text()
    return con.execute(sql).fetchdf()
