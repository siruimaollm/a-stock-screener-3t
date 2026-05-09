# -*- coding: utf-8 -*-
"""检查数据状态 + 代码关键逻辑"""
import sqlite3, sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 50)
print("[1] 数据状态")
print("=" * 50)
con = sqlite3.connect("data/stock_data.db")
latest_daily_date = con.execute("SELECT MAX(date) FROM daily_data_hfq").fetchone()[0]
latest_check_start = con.execute(
    "SELECT DATE(?, '-30 days')",
    (latest_daily_date,),
).fetchone()[0]
for tbl in ["daily_data_hfq", "index_data"]:
    r = con.execute(f"SELECT COUNT(*), MIN(date), MAX(date) FROM {tbl}").fetchone()
    print(f"  {tbl}: {r[0]:,} 行  {r[1]} ~ {r[2]}")
r = con.execute("SELECT COUNT(*) FROM stock_info").fetchone()
print(f"  stock_info: {r[0]:,} 只股票")

print("\n  最新5日行情 (000001 平安银行):")
rows = con.execute(
    "SELECT date,open,high,low,close,volume FROM daily_data_hfq "
    "WHERE stock_code='000001' ORDER BY date DESC LIMIT 5"
).fetchall()
for row in rows:
    print(f"    {row}")

print("\n  近10交易日每日覆盖股票数:")
rows = con.execute(
    "SELECT date, COUNT(DISTINCT stock_code) as n "
    "FROM daily_data_hfq "
    "WHERE date >= ? "
    "GROUP BY date ORDER BY date DESC LIMIT 10"
    , (latest_check_start,)
).fetchall()
for row in rows:
    print(f"    {row[0]}: {row[1]:,} 只")
con.close()

print()
print("=" * 50)
print("[2] 代码关键路径检查")
print("=" * 50)

import yaml
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

# volatility_filter
try:
    from src.volatility_filter import passes_volatility
    import inspect
    sig = inspect.signature(passes_volatility)
    print(f"  [OK] passes_volatility 参数: {list(sig.parameters.keys())}")
except Exception as e:
    print(f"  [FAIL] volatility_filter: {e}")

# scoring
try:
    from src.scoring import score_stock, score_all, is_selected
    print("  [OK] scoring 导入正常")
except Exception as e:
    print(f"  [FAIL] scoring: {e}")

# indicators
try:
    from src.indicators import add_all_indicators
    print("  [OK] indicators 导入正常")
except Exception as e:
    print(f"  [FAIL] indicators: {e}")

# data_fetcher 读取测试
try:
    from src.data_fetcher import load_all_kline_sqlite, load_benchmark_kline, load_stock_info_sqlite
    df = load_all_kline_sqlite("data/stock_data.db", start_date=latest_check_start, end_date=latest_daily_date)
    print(f"  [OK] load_all_kline_sqlite: {len(df):,} 行, {df['code'].nunique()} 只, cols={list(df.columns)}")
    latest_index_date = sqlite3.connect("data/stock_data.db").execute("SELECT MAX(date) FROM index_data").fetchone()[0]
    bm = load_benchmark_kline("data/stock_data.db", index_code="000300",
                               start_date=latest_check_start, end_date=latest_index_date)
    print(f"  [OK] load_benchmark_kline: {len(bm)} 行, latest={bm['date'].max()}")
    meta = load_stock_info_sqlite("data/stock_data.db")
    print(f"  [OK] load_stock_info_sqlite: {len(meta)} 只")
except Exception as e:
    traceback.print_exc()
    print(f"  [FAIL] data_fetcher: {e}")

# 指标+打分单股测试
try:
    df1 = load_all_kline_sqlite("data/stock_data.db",
                                 start_date="2025-10-01", end_date=latest_daily_date)
    grp = df1[df1["code"] == "000001"].sort_values("date").reset_index(drop=True)
    df1_ind = add_all_indicators(
        grp,
        rsi_periods=cfg["indicators"]["rsi_periods"],
        kdj_period=cfg["indicators"]["kdj_period"],
        kdj_smooth=cfg["indicators"]["kdj_smooth"],
        boll_period=cfg["indicators"]["boll_period"],
        boll_std=cfg["indicators"]["boll_std"],
        atr_period=cfg["indicators"]["atr_period"],
    )
    last = df1_ind.iloc[-1]
    print(f"  [OK] 指标计算(000001): RSI6={last.get('rsi6',0):.1f}  J={last.get('J',0):.1f}  pct_b={last.get('pct_b',0):.3f}")
    s = score_stock(df1_ind)
    print(f"       打分: boll={s['boll_score']} rsi={s['rsi_score']} kdj={s['kdj_score']} total={s['total_score']}")
    print(f"       signals: {s.get('signals', [])}")
except Exception as e:
    traceback.print_exc()
    print(f"  [FAIL] 指标/打分: {e}")

# volatility filter 单股测试
try:
    from src.volatility_filter import passes_volatility
    from run import _build_vol_cfg
    vol_cfg = _build_vol_cfg(cfg)
    ok = passes_volatility(df1_ind, **vol_cfg)
    print(f"  [OK] passes_volatility(000001): {ok}")
except Exception as e:
    traceback.print_exc()
    print(f"  [FAIL] passes_volatility: {e}")

print()
print("检查完成。")
