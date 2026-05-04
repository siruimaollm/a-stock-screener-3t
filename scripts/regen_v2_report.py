# -*- coding: utf-8 -*-
"""从已生成的 CSV 重新渲染 v2 回测 HTML 报告（含股票名称）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from src.backtest_report import save_backtest_v2_html
from src.data_fetcher import load_stock_info_sqlite

csv_path = "output/backtest_v2_20260504_trades.csv"
html_path = "output/backtest_v2_20260504.html"

trades_df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"code": str})
# Re-pad codes to 6 digits (CSV may have stripped leading zeros)
trades_df["code"] = trades_df["code"].astype(str).str.zfill(6)
print(f"Loaded {len(trades_df)} trades from {csv_path}")
print(f"Sample codes: {trades_df['code'].head(5).tolist()}")

# 注入股票名称 / 行业
meta = load_stock_info_sqlite("data/stock_data.db")
trades_df["name"]     = trades_df["code"].map(lambda c: (meta.get(c, {}) or {}).get("name", ""))
trades_df["industry"] = trades_df["code"].map(lambda c: (meta.get(c, {}) or {}).get("industry", ""))

# 重新排序列
cols = list(trades_df.columns)
if "name" in cols:
    cols.remove("name"); cols.insert(cols.index("code") + 1, "name")
if "industry" in cols:
    cols.remove("industry"); cols.insert(cols.index("name") + 1, "industry")
trades_df = trades_df[cols]
trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"CSV updated with names/industry: {csv_path}")

# 从 trades 数据重算 metrics
import numpy as np
n = len(trades_df)
rets = trades_df["return"].dropna()
excs = trades_df["excess_return"].dropna() if "excess_return" in trades_df else pd.Series([])
win = (rets > 0).sum()
tp_pct  = trades_df["exit_reason"].str.startswith("止盈(+").sum()
tp_boll = (trades_df["exit_reason"] == "止盈(布林上轨)").sum()
sl_cnt  = (trades_df["exit_reason"] == "破位止损").sum()
to_cnt  = trades_df["exit_reason"].str.startswith("超时").sum()

metrics = {
    "total_trades":  n,
    "win_rate":      round(win / n, 4),
    "tp_rate":       round((tp_pct + tp_boll) / n, 4),
    "tp_pct_rate":   round(tp_pct / n, 4),
    "tp_boll_rate":  round(tp_boll / n, 4),
    "sl_rate":       round(sl_cnt / n, 4),
    "timeout_rate":  round(to_cnt / n, 4),
    "mean_return":   round(float(rets.mean()), 4),
    "median_return": round(float(rets.median()), 4),
    "max_return":    round(float(rets.max()), 4),
    "min_return":    round(float(rets.min()), 4),
    "mean_excess":   round(float(excs.mean()), 4) if len(excs) else np.nan,
    "exc_win_rate":  round(float((excs > 0).mean()), 4) if len(excs) else np.nan,
    "mean_hold_days": round(float(trades_df["hold_days"].mean()), 1),
    "take_profit":   0.10,
    "break_down_pct": 0.08,
    "max_hold":      252,
}

save_backtest_v2_html(
    metrics, trades_df, html_path,
    start_date="2021-01-04",
    end_date="2026-04-25",
    meta=meta,
)
print(f"\nHTML 已重新生成: {html_path}")
print(f"  文件大小: {os.path.getsize(html_path):,} bytes")
