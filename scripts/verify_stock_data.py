from src.data_fetcher import load_all_kline_sqlite, load_stock_info_sqlite, load_benchmark_kline

print("Loading kline...")
df = load_all_kline_sqlite("data/stock_data.db", table="daily_data_hfq",
                           start_date="2026-03-01", end_date="2026-04-25")
print(f"  kline: {len(df)} rows, {df['code'].nunique()} stocks")
print(f"  columns: {list(df.columns)}")
print(df.head(2))

print()
print("Loading benchmark (000300)...")
bm = load_benchmark_kline("data/stock_data.db", index_code="000300",
                           start_date="2026-01-01", end_date="2026-04-25")
print(f"  benchmark: {len(bm)} rows")
print(bm.head(3))

print()
print("Loading stock info...")
meta = load_stock_info_sqlite("data/stock_data.db")
sample = list(meta.items())[:3]
print(f"  total: {len(meta)} stocks")
print(f"  sample: {sample}")
