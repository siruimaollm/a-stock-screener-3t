"""
Layer 2 — 三指标共振评分体系 (满分 0-9 分)
按 "A股大幅波动股票筛选策略 BOLL × RSI × KDJ" 文档实现，并在以下两点上更严格:
  - threshold = 7 (仅入选强信号)
  - RSI 看涨底背离 = 必要条件 (不只是加分项)

每个指标贡献 0-3 分。三指标方向须一致（同为买入或同为卖出）方可叠加。
本实现仅做买入方向（超卖反转）。

============================================================
BOLL 评分 (0-3 分) — 基于 %B 和带宽信号
  3 分: %B < 0.05  且 带宽扩张 且 价格向上突破中轨
  2 分: %B < 0.1  且 反弹K线(收 > 开)
  1 分: 仅带宽扩张 且 %B 在 [0.1, 0.3] 之间
  0 分: %B 处于 [0.3, 0.7] 区间内横向震荡

============================================================
RSI 评分 (0-3 分) — 双周期联动 RSI6(短) + RSI12(中)
  3 分: RSI6 < 30 且 RSI12 < 30 且 出现看涨底背离 [必要条件]
  2 分: RSI6 < 30 且 RSI12 < 30 (双周期同时超卖，无背离)
  1 分: RSI6 < 30，RSI12 在 [30, 50] (仅短周期超卖)
  0 分: RSI6 >= 30 或 RSI12 > 50 (分歧无效)

  看涨底背离严格定义:
    1) 当前价格创近 20 日新低
    2) 当前 RSI6 高于近20日内最低 RSI6 (RSI 形成更高的低点)
    3) 形成时 RSI6 < 40
    4) 配合 BOLL %B < 0.15 效果最佳

============================================================
KDJ 评分 (0-3 分) — J 线极值为最高权重的领先信号
  3 分: J < 0  (理论超卖极限突破)
  2 分: J < 20 且 K 在近 3 根K线内上穿 D (低位金叉)
  1 分: K < 30 且 K 向上运动
  0 分: KDJ 各线在 [30, 70] 区间内震荡

============================================================
入选标准:
  total_score >= threshold (默认 7，仅强信号)
  三个维度都 >= 1 分 (共振)
  has_divergence == True   (RSI看涨底背离作为必要条件)
"""
import numpy as np
import pandas as pd


def _cross_above(series_a: pd.Series, series_b: pd.Series, lookback: int = 3) -> bool:
    """series_a 在最近 lookback 根 K 线内上穿 series_b。"""
    if len(series_a) < lookback + 1:
        return False
    for i in range(-lookback, 0):
        prev_i = i - 1
        if (series_a.iloc[prev_i] <= series_b.iloc[prev_i] and
                series_a.iloc[i] > series_b.iloc[i]):
            return True
    return False


def _bw_expanding(df: pd.DataFrame, lookback: int = 5) -> bool:
    """当前带宽 > lookback 日前带宽 → 带宽由近期低点开始上翻。"""
    if "bb_width" not in df.columns or len(df) < lookback + 1:
        return False
    bw_now = df["bb_width"].iloc[-1]
    bw_past = df["bb_width"].iloc[-1 - lookback]
    if pd.isna(bw_now) or pd.isna(bw_past) or bw_past <= 0:
        return False
    return bw_now > bw_past


def detect_bullish_divergence(close: pd.Series, rsi: pd.Series,
                                lookback: int = 20) -> bool:
    """
    严格定义看涨底背离:
      1) 当前价格创近 lookback 日新低 (今日 close < 过去 lookback 日内除今日外的最低 close)
      2) 当前 RSI > 过去 lookback 日内最低 RSI (RSI 形成更高的低点)
      3) 形成时 RSI < 40 (PDF: 有效区间)
    """
    if len(close) < lookback + 1 or len(rsi) < lookback + 1:
        return False

    curr_close = close.iloc[-1]
    curr_rsi = rsi.iloc[-1]
    if pd.isna(curr_close) or pd.isna(curr_rsi):
        return False

    # 过去 lookback 日内除今日外的窗口
    past_close = close.iloc[-(lookback + 1):-1]
    past_rsi = rsi.iloc[-(lookback + 1):-1]
    if past_close.empty or past_rsi.empty:
        return False

    past_close_min = past_close.min()
    past_rsi_min = past_rsi.min()
    if pd.isna(past_close_min) or pd.isna(past_rsi_min):
        return False

    # 1) 今日价格创新低
    if curr_close >= past_close_min:
        return False

    # 2) 今日 RSI 高于过去窗口最低 RSI (RSI 未创新低 = 背离)
    if curr_rsi <= past_rsi_min:
        return False

    # 3) 当前 RSI < 40 才有效
    if curr_rsi >= 40:
        return False

    return True


def score_boll(df: pd.DataFrame) -> tuple:
    """Returns (score 0-3, list of triggered signals)."""
    if len(df) < 6:
        return 0, []

    last = df.iloc[-1]
    close = float(last["close"]) if not pd.isna(last.get("close", np.nan)) else None
    open_p = float(last["open"]) if "open" in df.columns and not pd.isna(last.get("open", np.nan)) else None
    bb_mid = float(last["bb_mid"]) if not pd.isna(last.get("bb_mid", np.nan)) else None
    pct_b = last.get("pct_b", np.nan)

    if close is None or pd.isna(pct_b):
        return 0, []

    bw_expanding = _bw_expanding(df, lookback=5)

    # 价格上穿中轨：今日 close > bb_mid 且 昨日 close <= 昨日 bb_mid
    crossed_mid = False
    if bb_mid is not None and close > bb_mid and len(df) >= 2:
        prev = df.iloc[-2]
        prev_close = float(prev["close"]) if not pd.isna(prev["close"]) else None
        prev_mid = float(prev["bb_mid"]) if not pd.isna(prev["bb_mid"]) else None
        if prev_close is not None and prev_mid is not None and prev_close <= prev_mid:
            crossed_mid = True

    bullish_candle = open_p is not None and close > open_p

    if pct_b < 0.05 and bw_expanding and crossed_mid:
        return 3, ["%B<0.05+带宽扩张+突破中轨"]
    if pct_b < 0.1 and bullish_candle:
        return 2, ["下轨阳线反弹(%B<0.1)"]
    if 0.1 <= pct_b < 0.3 and bw_expanding:
        return 1, ["带宽扩张+下沿区域"]
    return 0, []


def score_rsi(df: pd.DataFrame, rsi_long_oversold: float = 30) -> tuple:
    """
    Returns (score 0-3, list of signals, has_divergence flag).
    使用 RSI6 (短周期) + RSI12 (中周期) 双周期联动。
    rsi_long_oversold: RSI12 视为"超卖"的阈值。PDF原文=30，波动剧烈时可放宽到35。
    """
    if "rsi6" not in df.columns or "rsi12" not in df.columns:
        return 0, [], False

    last = df.iloc[-1]
    rsi6 = last.get("rsi6", np.nan)
    rsi12 = last.get("rsi12", np.nan)
    if pd.isna(rsi6) or pd.isna(rsi12):
        return 0, [], False

    rsi6 = float(rsi6)
    rsi12 = float(rsi12)

    has_divergence = detect_bullish_divergence(
        df["close"].astype(float), df["rsi6"], lookback=20
    )

    if rsi6 >= 30:
        return 0, [], has_divergence
    if rsi12 > 50:
        return 0, [], has_divergence

    if rsi12 < rsi_long_oversold:
        if has_divergence:
            return 3, [f"RSI6&12双周期超卖+底背离(R12<{rsi_long_oversold})"], True
        return 2, [f"RSI6&12双周期超卖(R12<{rsi_long_oversold})"], False

    return 1, [f"仅RSI6超卖(RSI12={rsi12:.1f})"], has_divergence


def score_kdj(df: pd.DataFrame) -> tuple:
    """Returns (score 0-3, list of signals)."""
    if "J" not in df.columns or "K" not in df.columns or "D" not in df.columns:
        return 0, []

    last = df.iloc[-1]
    j_val = last.get("J", np.nan)
    k_val = last.get("K", np.nan)
    if pd.isna(j_val) or pd.isna(k_val):
        return 0, []

    j_val = float(j_val)
    k_val = float(k_val)

    if j_val < 0:
        return 3, ["J<0极度超卖"]
    if j_val < 20 and _cross_above(df["K"], df["D"], lookback=3):
        return 2, ["J<20低位金叉(K穿D)"]
    if k_val < 30 and len(df) >= 2:
        k_prev = df["K"].iloc[-2]
        if not pd.isna(k_prev) and k_val > k_prev:
            return 1, ["K<30低位上行"]
    return 0, []


def score_stock(df: pd.DataFrame, rsi_long_oversold: float = 30) -> dict:
    """对单只股票打分。"""
    result = {
        "boll_score": 0,
        "rsi_score": 0,
        "kdj_score": 0,
        "total_score": 0,
        "has_divergence": False,
        "signals": [],
        "rsi6": np.nan, "rsi12": np.nan, "rsi24": np.nan,
        "K": np.nan, "D": np.nan, "J": np.nan,
        "bb_upper": np.nan, "bb_mid": np.nan, "bb_lower": np.nan,
        "bb_width": np.nan, "pct_b": np.nan,
        "atr_pct": np.nan, "amp30": np.nan,
        "close": np.nan, "open": np.nan, "volume": np.nan,
    }

    if df is None or len(df) < 30:
        return result

    last = df.iloc[-1]
    for col in ["rsi6", "rsi12", "rsi24",
                "K", "D", "J",
                "bb_upper", "bb_mid", "bb_lower", "bb_width", "pct_b",
                "atr_pct", "amp30", "close", "open", "volume"]:
        if col in df.columns:
            result[col] = last[col]

    boll, sig_b = score_boll(df)
    rsi, sig_r, has_div = score_rsi(df, rsi_long_oversold=rsi_long_oversold)
    kdj, sig_k = score_kdj(df)

    result["boll_score"] = boll
    result["rsi_score"] = rsi
    result["kdj_score"] = kdj
    result["total_score"] = boll + rsi + kdj
    result["has_divergence"] = has_div
    result["signals"] = sig_b + sig_r + sig_k
    return result


def is_selected(scores: dict, threshold: int = 7,
                require_all_dimensions: bool = True,
                require_rsi_divergence: bool = True) -> bool:
    """
    入选标准：
      total_score >= threshold (默认 7，仅强信号)
      三维度都 >= 1 分 (共振)
      has_divergence == True (RSI看涨底背离必要条件)
    """
    if scores["total_score"] < threshold:
        return False
    if require_all_dimensions:
        if scores["boll_score"] < 1:
            return False
        if scores["rsi_score"] < 1:
            return False
        if scores["kdj_score"] < 1:
            return False
    if require_rsi_divergence and not scores.get("has_divergence", False):
        return False
    return True


def score_all(stock_data: dict, threshold: int = 7,
              require_all_dimensions: bool = True,
              require_rsi_divergence: bool = True,
              rsi_long_oversold: float = 30, **kwargs) -> list:
    selected = []
    for code, df in stock_data.items():
        s = score_stock(df, rsi_long_oversold=rsi_long_oversold)
        s["code"] = code
        if is_selected(s, threshold, require_all_dimensions, require_rsi_divergence):
            selected.append(s)
    selected.sort(key=lambda x: x["total_score"], reverse=True)
    return selected
