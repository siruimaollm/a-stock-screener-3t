"""
策略回测 v2 — 新出场规则 + 高性能 numpy 实现

出场规则：
  止盈（先到先得）：
    1. 收益率 ≥ take_profit（默认 +10%，按当日 high 触发，按 TP 价成交）
    2. 收盘价 ≥ 布林上轨（按当日 close 成交）
  止损 / 强制平仓：
    3. 收盘价 ≤ entry × (1 - break_down_pct)（默认 -8%，破位）
    4. 持有交易日数 ≥ max_hold（默认 252，1 年自然到期）

性能优化：
  - 全市场指标只算一次（add_all_indicators）
  - 每只股票预转换为 numpy 数组（高/低/收/布林上轨等）
  - 用 np.searchsorted 做 O(log N) 日期定位
  - 出场扫描在 numpy 数组上进行（≈100 倍于 pandas 切片）
  - 指标已预算 → 信号日只需切片打分（O(1) iloc[:idx]）
"""
from __future__ import annotations

from typing import Any, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from src.indicators import add_all_indicators
from src.volatility_filter import passes_volatility
from src.scoring import score_stock, is_selected


# ─────────────────────────────────────────────
#  numpy 单股出场模拟
# ─────────────────────────────────────────────

def _simulate_exit(
    buy_idx: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    bb_upper_arr: np.ndarray,
    take_profit: float,
    break_down_pct: float,
    max_hold: int,
    stop_loss_mode: str = "close",
    fail_exit_days: int = 0,
    fail_exit_min_return: float = 0.0,
) -> Optional[Tuple[int, float, str]]:
    """
    对单笔交易做前向扫描，返回 (exit_idx, exit_price, reason)。
    全部 numpy 操作，无 pandas 调用。

    入场：buy_idx 当日开盘价
    扫描：buy_idx+1 ... min(buy_idx+max_hold, len-1)
    """
    n = close_arr.size
    if buy_idx >= n - 1:
        return None
    buy_price = open_arr[buy_idx]
    if not np.isfinite(buy_price) or buy_price <= 0:
        # 退化：用收盘
        buy_price = close_arr[buy_idx]
        if not np.isfinite(buy_price) or buy_price <= 0:
            return None

    tp_price = buy_price * (1.0 + take_profit)
    sl_price = buy_price * (1.0 - break_down_pct)

    end_idx = min(buy_idx + max_hold, n - 1)

    for i in range(buy_idx + 1, end_idx + 1):
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]
        bbu = bb_upper_arr[i] if bb_upper_arr is not None else np.nan

        if not np.isfinite(c):
            continue

        # 1. 破位止损
        if stop_loss_mode == "intraday":
            if np.isfinite(l) and l <= sl_price:
                return (i, float(sl_price), "破位止损(日内)")
        elif np.isfinite(c) and c <= sl_price:
            return (i, float(c), "破位止损")

        # 2. 固定止盈：当日 high 突破 tp_price
        if np.isfinite(h) and h >= tp_price:
            return (i, float(tp_price), "止盈(+%.0f%%)" % (take_profit * 100))

        # 3. 布林上轨止盈（按收盘判定）
        if np.isfinite(bbu) and c >= bbu:
            return (i, float(c), "止盈(布林上轨)")

        # 4. 短期失效退出：持有达到指定天数仍未达最低收益，按收盘离场
        if fail_exit_days > 0 and (i - buy_idx) >= fail_exit_days:
            curr_ret = (c - buy_price) / buy_price
            if curr_ret <= fail_exit_min_return:
                return (i, float(c), "失效退出(%dd)" % fail_exit_days)

    # 5. 超时（持有满 max_hold）
    last_close = close_arr[end_idx]
    if not np.isfinite(last_close) or last_close <= 0:
        return None
    return (end_idx, float(last_close), "超时(%dd)" % max_hold)


# ─────────────────────────────────────────────
#  信号日（每两周第一个交易日，避免每天扫描）
# ─────────────────────────────────────────────

def _signal_dates(trade_dates: List[str], freq: str = "biweekly") -> List[str]:
    """信号日采样：weekly / biweekly / monthly / daily。"""
    if freq == "daily":
        return list(trade_dates)
    signals, seen = [], set()
    for d in trade_dates:
        y, m, day = int(d[:4]), int(d[5:7]), int(d[8:10])
        if freq == "monthly":
            key = (y, m)
        elif freq == "weekly":
            # 简化：每周一/五个交易日一组
            from datetime import date as _date
            iso = _date(y, m, day).isocalendar()
            key = (iso.year, iso.week)
        else:  # biweekly
            key = (y, m, 1 if day <= 15 else 2)
        if key not in seen:
            signals.append(d)
            seen.add(key)
    return signals


# ─────────────────────────────────────────────
#  主回测函数
# ─────────────────────────────────────────────

def run_backtest_v2(
    all_kline: pd.DataFrame,
    benchmark_kline: pd.DataFrame,
    start_date: str,
    end_date: str,
    top_n: int = 20,
    vol_cfg: Optional[dict] = None,
    score_cfg: Optional[dict] = None,
    ind_cfg: Optional[dict] = None,
    take_profit: float = 0.10,
    break_down_pct: float = 0.08,
    max_hold: int = 252,
    signal_freq: str = "biweekly",
    min_history: int = 60,
    max_concurrent: int = 0,  # 0 = 不限并发
    stop_loss_mode: str = "close",
    fail_exit_days: int = 0,
    fail_exit_min_return: float = 0.0,
    min_candle_score: int = 0,
    min_vol_score: int = 0,
    min_total_score: int | None = None,
    require_rsi_or_kdj_score_sum: int = 0,
) -> Dict[str, Any]:
    """
    高性能回测 v2。出场规则见模块 docstring。

    返回:
      trades: pd.DataFrame  逐笔交易明细
      metrics: dict         汇总指标
    """
    if vol_cfg is None:    vol_cfg = {}
    if score_cfg is None:  score_cfg = {}
    if ind_cfg is None:    ind_cfg = {}

    threshold              = score_cfg.get("threshold", 9)
    require_all_dimensions = score_cfg.get("require_all_dimensions", True)
    require_rsi_divergence = score_cfg.get("require_rsi_divergence", False)
    rsi_long_oversold      = score_cfg.get("rsi_long_oversold", 40)
    strong_confirmation_threshold = score_cfg.get("strong_confirmation_threshold")
    weak_confirmation_threshold = score_cfg.get("weak_confirmation_threshold")
    score_min_candle = score_cfg.get("min_candle_score", 0)
    score_min_vol = score_cfg.get("min_vol_score", 0)
    score_min_total = score_cfg.get("min_total_score")
    score_rsi_kdj_sum = score_cfg.get("require_rsi_or_kdj_score_sum", 0)

    print(f"[v2] 回测区间: {start_date} ~ {end_date}")
    print(f"[v2] 出场规则: TP=+{take_profit*100:.0f}% 或 触布林上轨 | SL({stop_loss_mode})=-{break_down_pct*100:.0f}% | 最长 {max_hold} 日")
    if fail_exit_days > 0:
        print(f"[v2] 失效退出: 持有≥{fail_exit_days}日且收益≤{fail_exit_min_return:.1%}")

    # ── 1. 全市场交易日（用于信号日采样）──────────────────────
    trading_dates = sorted(all_kline["date"].unique())
    bt_dates = [d for d in trading_dates if start_date <= d <= end_date]
    sig_dates = _signal_dates(bt_dates, freq=signal_freq)
    print(f"[v2] 信号日: {len(sig_dates)} 个 (freq={signal_freq})")

    # ── 2. 基准（沪深300）────────────────────────────────────
    bench_map = {}
    if not benchmark_kline.empty:
        bench_map = dict(zip(benchmark_kline["date"],
                             benchmark_kline["close"].astype(float)))

    # ── 3. 预分组 + 预算指标 + 预转 numpy ──────────────────────
    print("[v2] 预分组数据...")
    by_code: Dict[str, pd.DataFrame] = {}
    for code, sub in all_kline.groupby("code"):
        sub = sub.sort_values("date").reset_index(drop=True)
        if len(sub) >= 30:
            by_code[code] = sub
    print(f"[v2] 共 {len(by_code)} 只股票")

    print("[v2] 预算指标 + numpy 转换 ...")
    indicator_cache: Dict[str, pd.DataFrame] = {}
    arr_cache: Dict[str, dict] = {}            # code -> {dates, open, high, low, close, bb_upper}
    for code, sub in by_code.items():
        try:
            df_ind = add_all_indicators(sub, **ind_cfg)
        except Exception:
            continue
        indicator_cache[code] = df_ind
        arr_cache[code] = {
            "dates":   df_ind["date"].values,
            "open":    df_ind["open"].astype(np.float64).values,
            "high":    df_ind["high"].astype(np.float64).values,
            "low":     df_ind["low"].astype(np.float64).values,
            "close":   df_ind["close"].astype(np.float64).values,
            "bb_upper": df_ind["bb_upper"].astype(np.float64).values
                        if "bb_upper" in df_ind.columns else None,
        }
    print(f"[v2] 指标 + numpy 完成 ({len(indicator_cache)} 只)")

    # 全市场交易日 → 索引
    td_to_idx = {d: i for i, d in enumerate(trading_dates)}

    # ── 4. 持仓管理（max_concurrent）──────────────────────────
    open_positions: Dict[str, Tuple[int, str]] = {}  # code -> (entry_global_idx, sig_date)
    trades: List[dict] = []

    # ── 5. 信号日循环 ────────────────────────────────────────
    print(f"[v2] 开始遍历 {len(sig_dates)} 个信号日 ...")
    for si, sig_date in enumerate(sig_dates):
        if si % 20 == 0:
            print(f"  {sig_date} ({si}/{len(sig_dates)}) trades={len(trades)} open={len(open_positions)}")

        sig_idx = td_to_idx[sig_date]
        if sig_idx + 1 >= len(trading_dates):
            continue
        buy_date = trading_dates[sig_idx + 1]
        buy_g_idx = sig_idx + 1

        # 释放已平仓的持仓（基于全局索引检查）
        # 出场扫描在加仓时就已计算，此处仅删除已结清条目
        # 简化做法：每次循环时只入场，平仓在最后统一扫描

        # 5.1 候选打分
        candidates = []
        for code, df_full in indicator_cache.items():
            arr = arr_cache.get(code)
            if arr is None:
                continue
            # 使用 searchsorted 找截至 sig_date 的位置
            idx = int(np.searchsorted(arr["dates"], sig_date, side="right"))
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
            if not is_selected(
                s,
                threshold,
                require_all_dimensions,
                require_rsi_divergence,
                strong_confirmation_threshold=strong_confirmation_threshold,
                weak_confirmation_threshold=weak_confirmation_threshold,
                min_candle_score=max(score_min_candle, min_candle_score),
                min_vol_score=max(score_min_vol, min_vol_score),
                min_total_score=min_total_score if min_total_score is not None else score_min_total,
                require_rsi_or_kdj_score_sum=max(score_rsi_kdj_sum, require_rsi_or_kdj_score_sum),
            ):
                continue
            candidates.append(s)

        candidates.sort(key=lambda x: x["total_score"], reverse=True)
        candidates = candidates[:top_n]

        # 5.2 对每只候选模拟出场（独立、不互相干扰）
        for p in candidates:
            code = p["code"]
            arr = arr_cache.get(code)
            if arr is None:
                continue
            local_buy_idx = int(np.searchsorted(arr["dates"], buy_date, side="left"))
            if local_buy_idx >= len(arr["dates"]):
                continue
            if arr["dates"][local_buy_idx] != buy_date:
                # 该股当日停牌或退市；跳过
                continue

            res = _simulate_exit(
                local_buy_idx,
                arr["open"], arr["high"], arr["low"], arr["close"],
                arr["bb_upper"],
                take_profit=take_profit,
                break_down_pct=break_down_pct,
                max_hold=max_hold,
                stop_loss_mode=stop_loss_mode,
                fail_exit_days=fail_exit_days,
                fail_exit_min_return=fail_exit_min_return,
            )
            if res is None:
                continue
            exit_idx, exit_price, reason = res
            buy_price = float(arr["open"][local_buy_idx])
            if not np.isfinite(buy_price) or buy_price <= 0:
                buy_price = float(arr["close"][local_buy_idx])
            if not np.isfinite(buy_price) or buy_price <= 0:
                continue

            ret = (exit_price - buy_price) / buy_price
            exit_date = str(arr["dates"][exit_idx])
            hold_days = exit_idx - local_buy_idx

            bench_buy  = bench_map.get(buy_date)
            bench_sell = bench_map.get(exit_date)
            if bench_buy and bench_sell and bench_buy > 0:
                bench_ret = (bench_sell - bench_buy) / bench_buy
                excess = ret - bench_ret
            else:
                bench_ret = np.nan
                excess = np.nan

            trades.append({
                "signal_date": sig_date,
                "buy_date":    buy_date,
                "exit_date":   exit_date,
                "code":        code,
                "buy_price":   round(buy_price, 3),
                "exit_price":  round(exit_price, 3),
                "return":      round(ret, 4),
                "bench_return": round(bench_ret, 4) if np.isfinite(bench_ret) else np.nan,
                "excess_return": round(excess, 4) if np.isfinite(excess) else np.nan,
                "hold_days":   int(hold_days),
                "exit_reason": reason,
                "score":       p["total_score"],
                "boll_score":  p["boll_score"],
                "rsi_score":   p["rsi_score"],
                "kdj_score":   p["kdj_score"],
                "vol_score":   p["vol_score"],
                "candle_score": p["candle_score"],
                "signals":     ";".join(p.get("signals", [])),
            })

    print(f"[v2] 总信号 {len(trades)} 笔")

    # ── 6. 汇总指标 ──────────────────────────────────────────
    if not trades:
        return {
            "trades": pd.DataFrame(),
            "metrics": {"total_trades": 0},
        }

    df_t = pd.DataFrame(trades)

    metrics = _aggregate_metrics(df_t, take_profit, break_down_pct, max_hold)
    return {"trades": df_t, "metrics": metrics}


def _aggregate_metrics(df: pd.DataFrame, take_profit: float,
                       break_down_pct: float, max_hold: int) -> dict:
    n = len(df)
    rets = df["return"].dropna()
    excs = df["excess_return"].dropna()

    win = (rets > 0).sum()
    tp_pct  = df["exit_reason"].str.startswith("止盈(+").sum()
    tp_boll = (df["exit_reason"] == "止盈(布林上轨)").sum()
    sl_cnt  = df["exit_reason"].str.startswith("破位止损").sum()
    to_cnt  = df["exit_reason"].str.startswith("超时").sum()
    fail_cnt = df["exit_reason"].str.startswith("失效退出").sum()

    return {
        "total_trades":  n,
        "win_rate":      round(win / n, 4),
        "tp_rate":       round((tp_pct + tp_boll) / n, 4),
        "tp_pct_rate":   round(tp_pct / n, 4),
        "tp_boll_rate":  round(tp_boll / n, 4),
        "sl_rate":       round(sl_cnt / n, 4),
        "timeout_rate":  round(to_cnt / n, 4),
        "fail_exit_rate": round(fail_cnt / n, 4),
        "mean_return":   round(float(rets.mean()), 4),
        "median_return": round(float(rets.median()), 4),
        "max_return":    round(float(rets.max()), 4),
        "min_return":    round(float(rets.min()), 4),
        "mean_excess":   round(float(excs.mean()), 4) if len(excs) else np.nan,
        "exc_win_rate":  round(float((excs > 0).mean()), 4) if len(excs) else np.nan,
        "mean_hold_days": round(float(df["hold_days"].mean()), 1),
        "take_profit":   take_profit,
        "break_down_pct": break_down_pct,
        "max_hold":      max_hold,
    }
