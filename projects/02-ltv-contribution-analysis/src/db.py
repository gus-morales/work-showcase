"""Thin DuckDB helper: loads the generated CSVs as tables and runs the
.sql files in sql/ against them, returning pandas DataFrames."""
from pathlib import Path

import duckdb
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SQL_DIR = BASE / "sql"
DATA_DIR = BASE / "data"


def get_connection():
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE customers AS SELECT * FROM read_csv_auto('{DATA_DIR / 'customers.csv'}')")
    con.execute(f"CREATE TABLE orders AS SELECT * FROM read_csv_auto('{DATA_DIR / 'orders.csv'}')")
    return con


def run_sql_file(con, filename: str) -> pd.DataFrame:
    sql = (SQL_DIR / filename).read_text()
    return con.execute(sql).fetchdf()
