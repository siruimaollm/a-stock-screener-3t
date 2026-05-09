import os
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_fetcher import (
    load_all_kline_sqlite,
    load_benchmark_kline,
    load_stock_info_sqlite,
)


DB_PATH = "data/stock_data.db"


def _latest_daily_date(db_path: str) -> str:
    con = sqlite3.connect(db_path)
    try:
        return con.execute("SELECT MAX(date) FROM daily_data_hfq").fetchone()[0]
    finally:
        con.close()


latest_date = _latest_daily_date(DB_PATH)
end_dt = datetime.strptime(latest_date, "%Y-%m-%d")
kline_start = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
bench_start = (end_dt - timedelta(days=120)).strftime("%Y-%m-%d")

print("Loading kline...")
df = load_all_kline_sqlite(
    DB_PATH,
    table="daily_data_hfq",
    start_date=kline_start,
    end_date=latest_date,
)
print(f"  kline: {len(df)} rows, {df['code'].nunique()} stocks")
print(f"  latest date: {latest_date}")
print(f"  columns: {list(df.columns)}")
print(df.head(2))

print()
print("Loading benchmark (000300)...")
bm = load_benchmark_kline(
    DB_PATH,
    index_code="000300",
    start_date=bench_start,
    end_date=latest_date,
)
print(f"  benchmark: {len(bm)} rows")
print(bm.head(3))

print()
print("Loading stock info...")
meta = load_stock_info_sqlite(DB_PATH)
sample = list(meta.items())[:3]
print(f"  total: {len(meta)} stocks")
print(f"  sample: {sample}")
