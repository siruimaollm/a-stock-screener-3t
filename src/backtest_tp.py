"""
简单止盈/止损回测（优化版）
  优化：指标只计算一次，所有信号日复用切片，速度提升 10-20x。

  - 信号日：每两周第一个交易日
  - 买入：次日开盘价
  - 卖出：+10%止盈 或 -5%止损，最长持有60交易日后按收盘价平仓
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def run_tp_backtest(all_kline: pd.DataFrame,
                    vol_cfg: dict,
                    score_cfg: dict,
                    ind_cfg: dict,
                    start_date: str,
                    end_date: str,
                    take_profit: float = 0.10,
                    stop_loss: float = 0.05,
                    max_hold: int = 60,
                    signal_freq: str = "biweekly") -> dict:

    from src.indicators import add_all_indicators
    from src.volatility_filter import passes_volatility
    from src.scoring import score_all

    all_dates = sorted(all_kline["date"].unique())
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]
    signal_dates = _get_signal_dates(trade_dates, freq=signal_freq)
    print(f"  回测区间: {start_date} ~ {end_date}  信号日: {len(signal_dates)} 个")

    # ── 优化核心：全量指标只算一次 ──────────────────────────
    print("  预计算全市场指标（仅一次）...")
    stock_data_full = {}
    for code, grp in all_kline.groupby("code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < 60:
            continue
        try:
            stock_data_full[code] = add_all_indicators(grp, **ind_cfg)
        except Exception:
            continue
    print(f"  指标预算完成，共 {len(stock_data_full)} 只股票")
    # ──────────────────────────────────────────────────────────

    trades = []
    for sig_date in signal_dates:
        # 对每个信号日，只取截至该日的切片
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
        )
        if not picks:
            continue

        future = [d for d in all_dates if d > sig_date]
        if not future:
            continue
        buy_date = future[0]

        for pick in picks:
            code = pick["code"]
            result = _simulate_trade(code, buy_date, all_kline,
                                     all_dates, take_profit, stop_loss, max_hold)
            if result is None:
                continue
            result.update({
                "signal_date": sig_date,
                "code": code,
                "total_score": pick["total_score"],
                "boll_score": pick["boll_score"],
                "rsi_score": pick["rsi_score"],
                "kdj_score": pick["kdj_score"],
                "vol_score": pick["vol_score"],
                "candle_score": pick["candle_score"],
                "has_divergence": pick["has_divergence"],
                "signals": ";".join(pick["signals"]),
            })
            trades.append(result)

    return _aggregate(trades, take_profit, stop_loss)


def _simulate_trade(code, buy_date, all_kline, all_dates,
                    take_profit, stop_loss, max_hold):
    stock = all_kline[all_kline["code"] == code].sort_values("date")
    buy_row = stock[stock["date"] == buy_date]
    if buy_row.empty:
        return None
    buy_price = float(buy_row["open"].iloc[0])
    if buy_price <= 0 or pd.isna(buy_price):
        return None

    tp_price = buy_price * (1 + take_profit)
    sl_price = buy_price * (1 - stop_loss)

    future_dates = [d for d in all_dates if d > buy_date][:max_hold]
    exit_date = exit_price = None
    exit_reason = "超时"

    for d in future_dates:
        row = stock[stock["date"] == d]
        if row.empty:
            continue
        high  = float(row["high"].iloc[0])
        low   = float(row["low"].iloc[0])
        close = float(row["close"].iloc[0])

        if low <= sl_price:
            exit_date, exit_price, exit_reason = d, sl_price, "止损"
            break
        if high >= tp_price:
            exit_date, exit_price, exit_reason = d, tp_price, "止盈"
            break
    else:
        if future_dates:
            last_d = future_dates[-1]
            row = stock[stock["date"] == last_d]
            if not row.empty:
                exit_date = last_d
                exit_price = float(row["close"].iloc[0])

    if exit_date is None or exit_price is None:
        return None

    ret = (exit_price - buy_price) / buy_price
    hold_days = len([d for d in all_dates if buy_date < d <= exit_date])
    return {
        "buy_date": buy_date, "buy_price": round(buy_price, 3),
        "exit_date": exit_date, "exit_price": round(exit_price, 3),
        "exit_reason": exit_reason,
        "return": round(ret, 4), "hold_days": hold_days,
    }


def _get_signal_dates(trade_dates, freq="biweekly"):
    signals, seen = [], set()
    for d in trade_dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        key = (dt.year, dt.month) if freq == "monthly" \
              else (dt.year, dt.month, 1 if dt.day <= 15 else 2)
        if key not in seen:
            signals.append(d)
            seen.add(key)
    return signals


def _aggregate(trades, take_profit, stop_loss):
    if not trades:
        return {"total_trades": 0, "message": "无有效交易"}
    df = pd.DataFrame(trades)
    total  = len(df)
    wins   = (df["return"] > 0).sum()
    tp_hits = (df["exit_reason"] == "止盈").sum()
    sl_hits = (df["exit_reason"] == "止损").sum()
    return {
        "total_trades":   total,
        "win_rate":       round(wins / total, 4),
        "tp_rate":        round(tp_hits / total, 4),
        "sl_rate":        round(sl_hits / total, 4),
        "timeout_rate":   round((total - tp_hits - sl_hits) / total, 4),
        "mean_return":    round(float(df["return"].mean()), 4),
        "median_return":  round(float(df["return"].median()), 4),
        "max_return":     round(float(df["return"].max()), 4),
        "min_return":     round(float(df["return"].min()), 4),
        "mean_hold_days": round(float(df["hold_days"].mean()), 1),
        "take_profit":    take_profit,
        "stop_loss":      stop_loss,
        "trades_df":      df,
    }
