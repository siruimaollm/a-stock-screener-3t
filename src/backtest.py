"""
Rolling backtest with optimized implementation.

For each trading day T in [start_date, end_date]:
  1. Compute signals using data up to T
  2. Select top_n stocks
  3. Simulate buy at T+1 open, hold hold_period trading days, sell at close

Benchmark: sh.000300 daily close return.
"""
import warnings
from typing import Any, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from src.indicators import add_all_indicators
from src.volatility_filter import passes_volatility
from src.scoring import score_stock, is_selected


warnings.filterwarnings("ignore")


def _trading_dates(all_kline: pd.DataFrame) -> List[str]:
    return sorted(all_kline["date"].unique().tolist())


def run_backtest(all_kline: pd.DataFrame,
                 benchmark_kline: pd.DataFrame,
                 start_date: str,
                 end_date: str,
                 hold_periods: List[int],
                 top_n: int = 20,
                 vol_cfg: Optional[dict] = None,
                 score_cfg: Optional[dict] = None,
                 ind_cfg: Optional[dict] = None,
                 min_history: int = 60) -> Dict[str, Any]:
    """
    Returns dict with keys:
      trades_{h}: DataFrame of all trades for hold period h
      metrics_{h}: dict of summary metrics for hold period h
      pass: bool (whether ALL pass criteria are met)
    """
    if vol_cfg is None:
        vol_cfg = {}
    if score_cfg is None:
        score_cfg = {}
    if ind_cfg is None:
        ind_cfg = {}

    threshold = score_cfg.get("threshold", 7)
    require_all_dimensions = score_cfg.get("require_all_dimensions", True)
    require_rsi_divergence = score_cfg.get("require_rsi_divergence", False)
    rsi_long_oversold = score_cfg.get("rsi_long_oversold", 40)

    trading_dates = _trading_dates(all_kline)
    # Precompute date → index lookup (O(1) lookups)
    date_to_tdx = {d: i for i, d in enumerate(trading_dates)}

    bench_map = {}
    if not benchmark_kline.empty:
        bench_map = dict(zip(benchmark_kline["date"], benchmark_kline["close"].astype(float)))

    # Filter to backtest window
    bt_dates = [d for d in trading_dates if start_date <= d <= end_date]

    # Pre-group kline by code, sort by date
    print("Pre-grouping kline data...")
    codes = all_kline["code"].unique().tolist()
    by_code = {}
    for c in codes:
        sub = all_kline[all_kline["code"] == c].sort_values("date").reset_index(drop=True)
        if len(sub) >= 30:
            by_code[c] = sub

    # Precompute all indicators (done once per stock)
    print(f"Computing indicators for {len(by_code)} stocks...")
    indicator_cache: dict = {}
    for code, df in by_code.items():
        try:
            indicator_cache[code] = add_all_indicators(df, **ind_cfg)
        except Exception:
            pass

    # Precompute per-stock: array of dates and fast open/close price lookups
    print("Building price lookup tables...")
    price_open: dict = {}   # code → {date: open_price}
    price_close: dict = {}  # code → {date: close_price}
    stock_dates: dict = {}  # code → sorted list of dates as strings
    stock_date_arr: dict = {}  # code → np.array of dates for searchsorted

    for code, df in by_code.items():
        dates_list = df["date"].tolist()
        stock_dates[code] = dates_list
        stock_date_arr[code] = np.array(dates_list)
        price_open[code] = dict(zip(df["date"], df["open"].astype(float)))
        price_close[code] = dict(zip(df["date"], df["close"].astype(float)))

    # collect trades per hold period
    trades: Dict[int, list] = {h: [] for h in hold_periods}

    print(f"Running backtest over {len(bt_dates)} trading days ({start_date} ~ {end_date})...")
    for i, t_date in enumerate(bt_dates):
        if i % 50 == 0:
            print(f"  {t_date} ({i}/{len(bt_dates)})  trades so far: {sum(len(v) for v in trades.values())}")

        t_idx = date_to_tdx[t_date]
        if t_idx + max(hold_periods) + 1 >= len(trading_dates):
            continue
        buy_date = trading_dates[t_idx + 1]

        # Score each stock using data up to t_date (inclusive)
        candidates = []
        for code, df_full in indicator_cache.items():
            # Fast: find row count up to t_date using searchsorted
            date_arr = stock_date_arr[code]
            # searchsorted returns insertion point; all indices < that are <= t_date
            idx = int(np.searchsorted(date_arr, t_date, side="right"))
            if idx < min_history:
                continue
            df_t = df_full.iloc[:idx]
            try:
                if not passes_volatility(df_t, **vol_cfg):
                    continue
                s = score_stock(df_t, rsi_long_oversold=rsi_long_oversold)
            except Exception:
                continue
            s["code"] = code
            if is_selected(s, threshold, require_all_dimensions, require_rsi_divergence):
                candidates.append(s)

        candidates.sort(key=lambda x: x["total_score"], reverse=True)
        top = candidates[:top_n]

        for p in top:
            code = p["code"]
            buy_price = price_open[code].get(buy_date)
            if buy_price is None or np.isnan(buy_price) or buy_price <= 0:
                buy_price = price_close[code].get(buy_date)
            if buy_price is None or np.isnan(buy_price) or buy_price <= 0:
                continue

            bench_buy = bench_map.get(buy_date)

            for h in hold_periods:
                sell_date_idx = t_idx + 1 + h
                if sell_date_idx >= len(trading_dates):
                    continue
                sell_date = trading_dates[sell_date_idx]
                sell_price = price_close[code].get(sell_date)
                if sell_price is None or np.isnan(sell_price) or sell_price <= 0:
                    continue

                ret = (sell_price - buy_price) / buy_price
                bench_sell = bench_map.get(sell_date)
                if bench_buy and bench_sell and bench_buy > 0:
                    bench_ret = (bench_sell - bench_buy) / bench_buy
                    excess = ret - bench_ret
                else:
                    bench_ret = np.nan
                    excess = np.nan

                trades[h].append({
                    "signal_date": t_date,
                    "buy_date": buy_date,
                    "sell_date": sell_date,
                    "code": code,
                    "buy_price": round(buy_price, 3),
                    "sell_price": round(sell_price, 3),
                    "return": round(ret, 4),
                    "bench_return": round(bench_ret, 4) if not np.isnan(bench_ret) else np.nan,
                    "excess_return": round(excess, 4) if not np.isnan(excess) else np.nan,
                    "score": p["total_score"],
                    "signals": ";".join(p.get("signals", [])),
                })

    results = {}
    all_pass = True

    for h in hold_periods:
        df_t = pd.DataFrame(trades[h])
        results[f"trades_{h}"] = df_t

        if df_t.empty:
            metrics = {"hold": h, "count": 0}
            results[f"metrics_{h}"] = metrics
            all_pass = False
            continue

        ret = df_t["return"].dropna()
        exc = df_t["excess_return"].dropna()

        win_rate = (ret > 0).mean()
        exc_win_rate = (exc > 0).mean() if len(exc) > 0 else np.nan
        mean_ret = ret.mean()
        mean_exc = exc.mean() if len(exc) > 0 else np.nan
        max_loss = ret.min()

        metrics = {
            "hold": h,
            "count": len(ret),
            "win_rate": round(win_rate, 4),
            "mean_return": round(mean_ret, 4),
            "median_return": round(ret.median(), 4),
            "max_loss": round(max_loss, 4),
            "exc_win_rate": round(float(exc_win_rate), 4) if not np.isnan(exc_win_rate) else np.nan,
            "mean_excess": round(float(mean_exc), 4) if not np.isnan(mean_exc) else np.nan,
        }
        results[f"metrics_{h}"] = metrics

    results["pass"] = all_pass
    results["trades"] = trades
    return results


def check_pass_criteria(results: dict, criteria: dict) -> Tuple[bool, List[str]]:
    """
    Returns (passed: bool, failed_messages: list[str]).
    """
    m10 = results.get("metrics_10", {})
    m20 = results.get("metrics_20", {})
    failed = []

    exc_wr = m10.get("exc_win_rate", 0) or 0
    if exc_wr < criteria.get("hold10_excess_win_rate", 0.55):
        failed.append(f"10日超额胜率 {exc_wr:.1%} < {criteria['hold10_excess_win_rate']:.1%}")

    exc_ret = m10.get("mean_excess", 0) or 0
    if exc_ret < criteria.get("hold10_excess_return_mean", 0.015):
        failed.append(f"10日平均超额收益 {exc_ret:.2%} < {criteria['hold10_excess_return_mean']:.2%}")

    ret20 = m20.get("mean_return", 0) or 0
    if ret20 < criteria.get("hold20_return_mean", 0.03):
        failed.append(f"20日平均收益 {ret20:.2%} < {criteria['hold20_return_mean']:.2%}")

    loss20 = m20.get("max_loss", 0) or 0
    if loss20 < criteria.get("hold20_max_loss", -0.25):
        failed.append(f"20日最大单笔亏损 {loss20:.2%} < {criteria['hold20_max_loss']:.2%}")

    for h in [5, 10, 20]:
        wr = results.get(f"metrics_{h}", {}).get("win_rate", 0) or 0
        if wr < criteria.get("any_win_rate", 0.50):
            failed.append(f"{h}日持有胜率 {wr:.1%} < {criteria['any_win_rate']:.1%}")

    return len(failed) == 0, failed


def build_monthly_heatmap(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Returns month x year pivot of mean return."""
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["ym"] = pd.to_datetime(df["buy_date"]).dt.to_period("M")
    df["year"] = pd.to_datetime(df["buy_date"]).dt.year
    df["month"] = pd.to_datetime(df["buy_date"]).dt.month
    piv = df.groupby(["year", "month"])["return"].mean().unstack("year").round(4)
    return piv
