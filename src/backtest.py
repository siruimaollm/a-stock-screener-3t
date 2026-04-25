"""
Rolling 3-year backtest.

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


def _get_price(all_kline: pd.DataFrame, code: str,
               date: str, price_col: str) -> Optional[float]:
    rows = all_kline[(all_kline["code"] == code) & (all_kline["date"] == date)]
    if rows.empty:
        return None
    val = rows.iloc[0][price_col]
    return float(val) if not pd.isna(val) else None


def run_backtest(all_kline: pd.DataFrame,
                 benchmark_kline: pd.DataFrame,
                 start_date: str,
                 end_date: str,
                 hold_periods: List[int],
                 top_n: int = 20,
                 vol_cfg: Optional[dict] = None,
                 score_cfg: Optional[dict] = None,
                 min_history: int = 120) -> Dict[str, Any]:
    """
    Returns dict with keys:
      trades_{h}: DataFrame of all trades for hold period h
      metrics_{h}: dict of summary metrics for hold period h
      pass: bool (whether ALL pass criteria are met)
      diagnostics: text summary
    """
    if vol_cfg is None:
        vol_cfg = {}
    if score_cfg is None:
        score_cfg = {}

    amp30_min = vol_cfg.get("amp30_min", 0.15)
    atr_pct_ratio = vol_cfg.get("atr_pct_ratio", 1.5)
    bb_width_quantile = vol_cfg.get("bb_width_quantile", 0.80)
    lookback = vol_cfg.get("lookback", 90)
    threshold = score_cfg.get("threshold", 7)
    require_all_dimensions = score_cfg.get("require_all_dimensions", True)
    require_rsi_divergence = score_cfg.get("require_rsi_divergence", True)
    rsi_long_oversold = score_cfg.get("rsi_long_oversold", 30)

    trading_dates = _trading_dates(all_kline)
    bench_map = {}
    if not benchmark_kline.empty:
        bench_map = dict(zip(benchmark_kline["date"], benchmark_kline["close"].astype(float)))

    # Filter to backtest window
    bt_dates = [d for d in trading_dates if start_date <= d <= end_date]

    # Pre-group kline by code
    codes = all_kline["code"].unique().tolist()
    by_code = {c: all_kline[all_kline["code"] == c].sort_values("date").reset_index(drop=True)
               for c in codes}

    # Precompute all indicators (expensive but done once)
    print("Computing indicators for all stocks...")
    indicator_cache: dict[str, pd.DataFrame] = {}
    for code, df in by_code.items():
        if len(df) >= 30:
            indicator_cache[code] = add_all_indicators(df)

    # collect trades per hold period
    trades: Dict[int, list] = {h: [] for h in hold_periods}

    print(f"Running backtest over {len(bt_dates)} trading days...")
    for i, t_date in enumerate(bt_dates):
        if i % 20 == 0:
            print(f"  {t_date} ({i}/{len(bt_dates)})")

        # find T+1 (buy date)
        t_idx = trading_dates.index(t_date)
        if t_idx + max(hold_periods) + 1 >= len(trading_dates):
            continue
        buy_date = trading_dates[t_idx + 1]

        # score stocks with data up to t_date
        candidates = []
        for code, df_full in indicator_cache.items():
            df_t = df_full[df_full["date"] <= t_date]
            if len(df_t) < min_history:
                continue
            if not passes_volatility(df_t, amp30_min, atr_pct_ratio, bb_width_quantile, lookback):
                continue
            s = score_stock(df_t, rsi_long_oversold=rsi_long_oversold)
            s["code"] = code
            if is_selected(s, threshold, require_all_dimensions, require_rsi_divergence):
                candidates.append(s)

        candidates.sort(key=lambda x: x["total_score"], reverse=True)
        top = candidates[:top_n]

        for p in top:
            code = p["code"]
            buy_price = _get_price(all_kline, code, buy_date, "open")
            if buy_price is None or buy_price <= 0:
                buy_price = _get_price(all_kline, code, buy_date, "close")
            if buy_price is None or buy_price <= 0:
                continue

            bench_buy = bench_map.get(buy_date)

            for h in hold_periods:
                sell_date_idx = t_idx + 1 + h
                if sell_date_idx >= len(trading_dates):
                    continue
                sell_date = trading_dates[sell_date_idx]
                sell_price = _get_price(all_kline, code, sell_date, "close")
                if sell_price is None or sell_price <= 0:
                    continue

                ret = (sell_price - buy_price) / buy_price
                bench_sell = bench_map.get(sell_date)
                bench_ret = ((bench_sell - bench_buy) / bench_buy
                             if bench_buy and bench_sell and bench_buy > 0 else np.nan)
                excess = ret - bench_ret if not np.isnan(bench_ret) else np.nan

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
    criteria keys from config.yaml backtest.pass_criteria.
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
