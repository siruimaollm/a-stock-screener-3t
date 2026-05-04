"""
Layer 1 — 波动率过滤（三段时序验证版）

设计目标：筛出「历史曾大涨放量、随后缩量横盘整理」的股票，
         避免选到「刚放量跌下来、尚未止跌横盘」的标的。

三段验证（全部 AND）：

  第一段 — 历史曾大涨放量（60~25日前窗口）
    1. 最大涨幅 ≥ 25%（曾被资金推升）
    2. 上涨段平均成交量 > 90日均量×1.3（放量推升，非阴跌反弹）
    3. 上涨持续 ≥ 5个交易日（排除单日异动）

  第二段 — 近期缩量横盘（最近30日）
    4. 近30日价格区间/均价 ≤ 12%（价格收敛，确认横盘）
    5. 当前价 ≥ 60日高点的 40%~80%（回撤受控，未崩盘）
    6. 近10日均量 < 近30日均量×0.6（缩量已成事实）
    7. 近10日ATR < 近30日ATR×0.75（波动率明显收敛）

  第三段 — 当前未在急跌中
    8. 近5日累计跌幅 > -10%（避开正在下跌的标的）

  基础门槛（同前）
    9. 收盘价 ≥ min_close（剔除暴雷/面值退市风险）
   10. 5日均成交额 ≥ min_avg_amount（流动性硬门槛）
"""
import numpy as np
import pandas as pd


def passes_volatility(
    df: pd.DataFrame,
    # 基础门槛
    min_close: float = 3.0,
    min_avg_amount: float = 5e7,
    # 第一段：历史曾大涨
    rally_lookback: int = 60,
    rally_min_pct: float = 0.25,
    rally_vol_ratio: float = 1.3,
    rally_min_days: int = 5,
    # 第二段：缩量横盘
    consol_window: int = 30,
    range_max_pct: float = 0.12,
    drawdown_min: float = 0.40,
    drawdown_max: float = 0.80,
    vol_shrink_ratio: float = 0.6,
    atr_shrink_ratio: float = 0.75,
    # 第三段：未在急跌
    cum_5d_min: float = -0.10,
    # 兼容旧接口（不再使用，保留避免调用方报错）
    amp30_min: float = 0.15,
    atr_pct_ratio: float = 1.2,
    bb_width_quantile: float = 0.65,
    lookback: int = 90,
    **kwargs,
) -> bool:
    """
    三段时序验证：历史大涨放量 → 近期缩量横盘 → 当前未急跌。
    df 须已包含 add_all_indicators() 输出的列。
    """
    if len(df) < 90:
        return False

    last = df.iloc[-1]

    # ── 基础门槛 ──────────────────────────────────────────────
    close = last.get("close")
    if pd.isna(close) or float(close) < min_close:
        return False

    amt_ma5 = last.get("amount_ma5")
    if pd.isna(amt_ma5) or float(amt_ma5) < min_avg_amount:
        return False

    close = float(close)

    # ── 第一段：历史曾大涨放量（rally_lookback~consol_window+5 日前）──
    # 用 [-rally_lookback : -(consol_window-5)] 作为"历史涨幅检测窗口"
    # 默认 df.iloc[-60:-25]，35天窗口
    rally_end = -(consol_window - 5)          # -25
    rally_df = df.iloc[-rally_lookback: rally_end]
    if len(rally_df) < 10:
        return False

    rally_high = float(rally_df["high"].max())
    rally_low  = float(rally_df["low"].min())
    if rally_low <= 0:
        return False

    rally_pct = (rally_high - rally_low) / rally_low
    if rally_pct < rally_min_pct:            # ① 涨幅不足25%
        return False

    # 上涨段平均成交量 vs 90日基准量
    vol_baseline = float(df["volume"].iloc[-90:].mean())
    vol_rally    = float(rally_df["volume"].mean())
    if vol_baseline <= 0 or vol_rally < vol_baseline * rally_vol_ratio:  # ② 放量不足
        return False

    # 上涨持续天数：从最低点到最高点的位置差
    lows  = rally_df["low"].values
    highs = rally_df["high"].values
    low_pos  = int(np.argmin(lows))
    # 在 low_pos 之后寻找最高点
    highs_after = highs[low_pos:]
    if len(highs_after) == 0:
        return False
    high_pos = int(np.argmax(highs_after)) + low_pos
    if high_pos - low_pos < rally_min_days:  # ③ 上涨持续天数不足
        return False

    # ── 第二段：近期缩量横盘（最近 consol_window 日）────────
    consol_df = df.iloc[-consol_window:]
    if len(consol_df) < 15:
        return False

    # ④ 价格区间收敛
    c_max  = float(consol_df["close"].max())
    c_min  = float(consol_df["close"].min())
    c_mean = float(consol_df["close"].mean())
    if c_mean <= 0:
        return False
    range_pct = (c_max - c_min) / c_mean
    if range_pct > range_max_pct:            # 价格区间超过12%，未横盘
        return False

    # ⑤ 当前价与60日高点的比值（40%~80%）
    high_60d = float(df["high"].iloc[-rally_lookback:].max())
    if high_60d <= 0:
        return False
    dist_ratio = close / high_60d
    if not (drawdown_min <= dist_ratio <= drawdown_max):
        return False

    # ⑥ 成交量萎缩：近10日均量 < 近30日均量×0.6
    vol_10d = float(df["volume"].iloc[-10:].mean())
    vol_30d = float(df["volume"].iloc[-30:].mean())
    if vol_30d <= 0 or vol_10d / vol_30d > vol_shrink_ratio:
        return False

    # ⑦ ATR萎缩：近10日ATR < 近30日ATR×0.75
    if "atr" in df.columns:
        atr_10d = float(df["atr"].iloc[-10:].mean())
        atr_30d = float(df["atr"].iloc[-30:].mean())
        if atr_30d > 0 and atr_10d / atr_30d > atr_shrink_ratio:
            return False

    # ── 第三段：当前未在急跌中 ───────────────────────────────
    # ⑧ 近5日累计跌幅 > -10%
    if len(df) >= 6:
        close_5d_ago = float(df["close"].iloc[-6])
        if close_5d_ago > 0:
            cum_5d = (close - close_5d_ago) / close_5d_ago
            if cum_5d < cum_5d_min:
                return False

    return True


def filter_universe(stock_data: dict, **kwargs) -> list:
    """Returns list of codes that pass volatility gate."""
    return [code for code, df in stock_data.items()
            if passes_volatility(df, **kwargs)]
