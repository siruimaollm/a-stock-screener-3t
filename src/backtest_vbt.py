"""
VectorBT 向量化止盈/止损回测（v2）

策略规则：
  - 信号日：每两周第一个交易日
  - 买入：次日开盘价，每期最多同时持有 max_positions 只
  - 止盈：收益达 +take_profit（默认10%）或 收盘价触及布林上轨（先到先得）
  - 止损：亏损达 -stop_loss（默认10%）
  - 超时：持有超过 max_hold 交易日后按收盘价平仓

加速亮点：
  - numba JIT KDJ：~50× 加速
  - VectorBT 向量化：全市场同时模拟，替代逐笔 Python 循环
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────
#  信号日
# ─────────────────────────────────────────────

def _get_signal_dates(trade_dates: list[str], freq: str = "biweekly") -> list[str]:
    if freq == "daily":
        return list(trade_dates)

    signals, seen = [], set()
    for d in trade_dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        if freq == "monthly":
            key = (dt.year, dt.month)
        elif freq == "weekly":
            iso = dt.isocalendar()
            key = (iso.year, iso.week)
        elif freq == "biweekly":
            key = (dt.year, dt.month, 1 if dt.day <= 15 else 2)
        else:
            raise ValueError(f"Unsupported signal_freq: {freq}")
        if key not in seen:
            signals.append(d)
            seen.add(key)
    return signals


# ─────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────

def run_vbt_backtest(
    all_kline: pd.DataFrame,
    vol_cfg: dict,
    score_cfg: dict,
    ind_cfg: dict,
    start_date: str,
    end_date: str,
    take_profit: float = 0.10,
    stop_loss: float = 0.10,
    max_hold: int = 60,
    max_positions: int = 5,
    signal_freq: str = "biweekly",
) -> dict:
    """
    向量化止盈止损回测（布林上轨 + 固定TP 双止盈）。
    """
    from src.indicators import add_all_indicators
    from src.volatility_filter import passes_volatility
    from src.scoring import score_all

    all_dates = sorted(all_kline["date"].unique())
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]
    signal_dates = _get_signal_dates(trade_dates, freq=signal_freq)
    print(f"  回测区间: {start_date} ~ {end_date}  信号日: {len(signal_dates)} 个")

    # ── 1. 预计算全市场指标（一次性）──────────────────────────
    print("  预计算全市场指标（numba KDJ 加速）...")
    stock_data_full: dict[str, pd.DataFrame] = {}
    for code, grp in all_kline.groupby("code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < 60:
            continue
        try:
            stock_data_full[code] = add_all_indicators(grp, **ind_cfg)
        except Exception:
            continue
    print(f"  指标预算完成，共 {len(stock_data_full)} 只股票")

    # ── 2. 逐信号日打分，每期取 top max_positions ───────────
    # entry_records: list of (buy_date, code, signal_date)
    entry_records: list[tuple[str, str, str]] = []

    for sig_date in signal_dates:
        stock_data = {}
        for code, df in stock_data_full.items():
            sub = df[df["date"] <= sig_date]
            if len(sub) < 60:
                continue
            stock_data[code] = sub

        filtered = {c: d for c, d in stock_data.items()
                    if passes_volatility(d, **vol_cfg)}

        picks = score_all(
            filtered,
            threshold=score_cfg.get("threshold", 9),
            require_all_dimensions=score_cfg.get("require_all_dimensions", True),
            require_rsi_divergence=score_cfg.get("require_rsi_divergence", False),
            rsi_long_oversold=score_cfg.get("rsi_long_oversold", 40),
            strong_confirmation_threshold=score_cfg.get("strong_confirmation_threshold"),
            weak_confirmation_threshold=score_cfg.get("weak_confirmation_threshold"),
            min_candle_score=score_cfg.get("min_candle_score", 0),
            min_vol_score=score_cfg.get("min_vol_score", 0),
            min_total_score=score_cfg.get("min_total_score"),
            require_rsi_or_kdj_score_sum=score_cfg.get("require_rsi_or_kdj_score_sum", 0),
        )
        if not picks:
            continue

        # 每期最多 max_positions 只
        picks = picks[:max_positions]

        future = [d for d in all_dates if d > sig_date]
        if not future:
            continue
        buy_date = future[0]

        for pick in picks:
            entry_records.append((buy_date, pick["code"], sig_date))

    total_signals = len(entry_records)
    print(f"  共 {total_signals} 条入场信号（每期≤{max_positions}只）")

    if not entry_records:
        return {"total_trades": 0, "message": "无有效交易"}

    # ── 3. 构建价格矩阵（只保留有信号的股票）──────────────
    signal_codes = list({code for _, code, _ in entry_records})

    # 取整个回测区间 + 前置历史（指标预热）
    hist_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
    sub = all_kline[
        all_kline["code"].isin(signal_codes) &
        (all_kline["date"] >= hist_start) &
        (all_kline["date"] <= end_date)
    ].copy()

    def _pivot(col: str) -> pd.DataFrame:
        return sub.pivot(index="date", columns="code", values=col).sort_index()

    close_p = _pivot("close")
    open_p  = _pivot("open")
    high_p  = _pivot("high")
    low_p   = _pivot("low")
    dates_idx = close_p.index.tolist()

    # ── 4. 布林上轨矩阵（用于额外止盈信号）────────────────
    # 从预计算的指标数据中提取 bb_upper
    bb_upper_records = []
    for code in signal_codes:
        if code not in stock_data_full:
            continue
        df = stock_data_full[code]
        if "bb_upper" not in df.columns:
            continue
        tmp = df[["date", "bb_upper"]].copy()
        tmp["code"] = code
        bb_upper_records.append(tmp)

    if bb_upper_records:
        bb_df = pd.concat(bb_upper_records, ignore_index=True)
        bb_upper_p = bb_df.pivot(index="date", columns="code", values="bb_upper")
        # 对齐到 close_p 的索引
        bb_upper_p = bb_upper_p.reindex(index=close_p.index, columns=close_p.columns)
    else:
        bb_upper_p = pd.DataFrame(np.nan, index=close_p.index, columns=close_p.columns)

    # ── 5. 构建 entries / exits 矩阵 ────────────────────
    entries = pd.DataFrame(False, index=close_p.index, columns=close_p.columns)
    timeout_exits = pd.DataFrame(False, index=close_p.index, columns=close_p.columns)

    for buy_date, code, _ in entry_records:
        if buy_date in entries.index and code in entries.columns:
            entries.loc[buy_date, code] = True
        if buy_date in dates_idx and code in timeout_exits.columns:
            buy_idx = dates_idx.index(buy_date)
            exit_idx = min(buy_idx + max_hold, len(dates_idx) - 1)
            timeout_exits.loc[dates_idx[exit_idx], code] = True

    # 布林上轨止盈：close >= bb_upper（用作额外的 exits 信号，VectorBT 会在入场后处理）
    bb_tp_exits = (close_p >= bb_upper_p).fillna(False)

    # 合并超时 + 布林上轨退出
    exits = timeout_exits | bb_tp_exits

    # ── 6. VectorBT 向量化回测 ──────────────────────────
    print("  运行 VectorBT 向量化回测...")
    pf = vbt.Portfolio.from_signals(
        close=close_p,
        open=open_p,
        high=high_p,
        low=low_p,
        entries=entries,
        exits=exits,
        sl_stop=stop_loss,       # 固定止损（-10%）
        tp_stop=take_profit,     # 固定止盈（+10%）
        upon_stop_exit="close",  # 触发当日收盘平仓
        stop_entry_price=2,      # FillPrice = 开盘成交价
        freq="1D",
        init_cash=1_000_000,
    )

    # ── 7. 整理结果 ──────────────────────────────────────
    trades = pf.trades.records_readable
    if trades.empty:
        return {"total_trades": 0, "message": "无成交记录"}

    trades = trades.rename(columns={
        "Column":          "code",
        "Entry Timestamp": "buy_date",
        "Exit Timestamp":  "exit_date",
        "Avg Entry Price": "buy_price",
        "Avg Exit Price":  "exit_price",
        "Return":          "return",
    })

    # 持有天数
    try:
        trades["hold_days"] = (
            pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["buy_date"])
        ).dt.days
    except Exception:
        trades["hold_days"] = 0

    # 推断退出原因
    def _infer_reason(row):
        try:
            ep = float(row["buy_price"])
            xp = float(row["exit_price"])
            ret = (xp - ep) / ep if ep > 0 else 0
            if ret <= -stop_loss * 0.95:
                return "止损"
            if ret >= take_profit * 0.95:
                return "止盈(+10%)"
            # 检查是否触及布林上轨（收益为正但未达10%）
            if ret > 0:
                return "止盈(布林上轨)"
        except Exception:
            pass
        return "超时"

    trades["exit_reason"] = trades.apply(_infer_reason, axis=1)

    return _aggregate(trades, take_profit, stop_loss)


def _aggregate(trades: pd.DataFrame, take_profit: float, stop_loss: float) -> dict:
    df = trades
    total   = len(df)
    wins    = (df["return"] > 0).sum()
    tp_hits      = (df["exit_reason"].str.startswith("止盈")).sum()
    tp_pct_hits  = (df["exit_reason"] == "止盈(+10%)").sum()
    tp_boll_hits = (df["exit_reason"] == "止盈(布林上轨)").sum()
    sl_hits      = (df["exit_reason"] == "止损").sum()

    return {
        "total_trades":    total,
        "win_rate":        round(wins / total, 4),
        "tp_rate":         round(tp_hits / total, 4),
        "tp_pct_rate":     round(tp_pct_hits / total, 4),
        "tp_boll_rate":    round(tp_boll_hits / total, 4),
        "sl_rate":         round(sl_hits / total, 4),
        "timeout_rate":    round((total - tp_hits - sl_hits) / total, 4),
        "mean_return":     round(float(df["return"].mean()), 4),
        "median_return":   round(float(df["return"].median()), 4),
        "max_return":      round(float(df["return"].max()), 4),
        "min_return":      round(float(df["return"].min()), 4),
        "mean_hold_days":  round(float(df["hold_days"].mean()), 1),
        "take_profit":     take_profit,
        "stop_loss":       stop_loss,
        "trades_df":       df,
    }
