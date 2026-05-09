"""
Layer 1 — 波动率过滤（三段时序验证版 · 优化）

设计目标：筛出「历史曾活跃、近期缩量整理、未在急跌中」的股票，
         作为 Layer 2 五维评分的候选池。

性能优化：
  - 使用 numpy 数组运算替代 pandas 切片（提速 5-10x）
  - 早退出（fail-fast）：基础门槛先于复杂检查

三段验证（全部 AND）：

  第一段 — 历史曾活跃（rally_lookback ~ consol_window+5 日前）
    1. 最大涨幅 ≥ rally_min_pct（曾被资金推升）
    2. 上涨段平均成交量 > 90日均量×rally_vol_ratio（放量推升）
    3. 上涨持续 ≥ rally_min_days 个交易日

  第二段 — 近期缩量整理（最近 consol_window 日）
    4. 价格区间/均价 ≤ range_max_pct（价格收敛，已横盘）
    5. 当前价 ≥ 60日高点的 drawdown_min ~ drawdown_max（回撤受控）
    6. 近10日均量 < 更长周期均量（默认 60 日，方向性缩量）
    7. 近10日 ATR% < 近30日 ATR%×atr_shrink_ratio（波动率收敛）

  第三段 — 当前未急跌
    8. 近5日累计涨跌幅 > cum_5d_min（避开正在下跌的标的）

  基础门槛
    9. 收盘价 ≥ min_close
   10. 5日均成交额 ≥ min_avg_amount
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
    range_max_pct: float = 0.20,
    drawdown_min: float = 0.40,
    drawdown_max: float = 0.90,
    directional_vol_shrink: bool = True,
    vol_shrink_short_window: int = 10,
    vol_shrink_long_window: int = 60,
    vol_shrink_ratio: float = 0.80,
    atr_shrink_ratio: float = 0.85,
    # 第三段：未在急跌
    cum_5d_min: float = -0.10,
    # 兼容旧接口（保留避免调用方报错）
    amp30_min: float = 0.15,
    atr_pct_ratio: float = 1.2,
    bb_width_quantile: float = 0.65,
    lookback: int = 90,
    **kwargs,
) -> bool:
    """
    三段时序验证。df 须已包含 add_all_indicators() 输出的列。
    返回 True 表示通过过滤。
    """
    n = len(df)
    if n < 90:
        return False

    # ── 基础门槛（最先 fail-fast）─────────────────────────────
    # 直接用 numpy 数组访问最后一行，避免 .iloc[-1] 创建 Series
    close_arr = df["close"].values
    close = close_arr[-1]
    if not np.isfinite(close) or close < min_close:
        return False

    amt_ma5_col = df["amount_ma5"].values if "amount_ma5" in df.columns else None
    if amt_ma5_col is None or not np.isfinite(amt_ma5_col[-1]) or amt_ma5_col[-1] < min_avg_amount:
        return False

    high_arr = df["high"].values
    low_arr  = df["low"].values
    vol_arr  = df["volume"].values

    # ── 第三段优先（最容易 fail-fast，避免无谓的Phase 1计算）──
    if n >= 6:
        c5_ago = close_arr[-6]
        if c5_ago > 0:
            cum_5d = (close - c5_ago) / c5_ago
            if cum_5d < cum_5d_min:
                return False

    # ── 第一段：历史曾活跃 ──────────────────────────────────
    rally_end = -(consol_window - 5)        # -25 (默认)
    rally_start = -rally_lookback           # -60
    if n + rally_start < 0:
        return False
    rally_high = high_arr[rally_start:rally_end]
    rally_low  = low_arr [rally_start:rally_end]
    rally_vol  = vol_arr [rally_start:rally_end]
    if rally_high.size < 10:
        return False

    r_high = rally_high.max()
    r_low  = rally_low.min()
    if r_low <= 0:
        return False
    rally_pct = (r_high - r_low) / r_low
    if rally_pct < rally_min_pct:
        return False

    vol_baseline = vol_arr[-90:].mean()
    if vol_baseline <= 0 or rally_vol.mean() < vol_baseline * rally_vol_ratio:
        return False

    # 上涨持续天数（从最低位到最高位的索引差）
    low_pos = int(rally_low.argmin())
    highs_after = rally_high[low_pos:]
    if highs_after.size == 0:
        return False
    high_pos = int(highs_after.argmax()) + low_pos
    if high_pos - low_pos < rally_min_days:
        return False

    # ── 第二段：缩量整理 ────────────────────────────────────
    consol_close = close_arr[-consol_window:]
    if consol_close.size < 15:
        return False
    c_max  = consol_close.max()
    c_min  = consol_close.min()
    c_mean = consol_close.mean()
    if c_mean <= 0:
        return False
    if (c_max - c_min) / c_mean > range_max_pct:
        return False

    high_60d = high_arr[-rally_lookback:].max()
    if high_60d <= 0:
        return False
    dist_ratio = close / high_60d
    if not (drawdown_min <= dist_ratio <= drawdown_max):
        return False

    short_window = max(int(vol_shrink_short_window), 1)
    long_window = max(int(vol_shrink_long_window), short_window + 1)
    if n < long_window:
        return False

    vol_short = vol_arr[-short_window:].mean()
    vol_long = vol_arr[-long_window:].mean()
    if vol_long <= 0:
        return False
    if directional_vol_shrink:
        if vol_short >= vol_long:
            return False
    elif vol_short / vol_long > vol_shrink_ratio:
        return False

    # ATR 收敛检查（用 atr_pct，与 indicators.py 一致）
    if "atr_pct" in df.columns:
        atr_arr = df["atr_pct"].values
        atr_10 = np.nanmean(atr_arr[-10:])
        atr_30 = np.nanmean(atr_arr[-30:])
        if atr_30 > 0 and atr_10 / atr_30 > atr_shrink_ratio:
            return False

    return True


def filter_universe(stock_data: dict, **kwargs) -> list:
    """Returns list of codes that pass volatility gate."""
    return [code for code, df in stock_data.items()
            if passes_volatility(df, **kwargs)]
