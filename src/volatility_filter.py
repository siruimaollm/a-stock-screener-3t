"""
Layer 1 — 30-day volatility filter (PDF "第一层：波动幅度筛选").

All three conditions must pass to enter the scoring layer:
  1. 30日最大涨跌幅绝对值 ≥ amp30_min   (default 15%)
  2. ATR%当前值 > 近90日均值 × atr_pct_ratio   (default 1.5×, 真实波动正在扩张)
  3. BB带宽 处于近90日 bb_width_quantile 分位以上   (default 80%, 价格波动空间充足)
"""
import pandas as pd


def passes_volatility(df: pd.DataFrame,
                      amp30_min: float = 0.15,
                      atr_pct_ratio: float = 1.5,
                      bb_width_quantile: float = 0.80,
                      lookback: int = 90) -> bool:
    """
    Check whether the LAST row of df passes all three volatility gates.
    df must already have indicator columns appended by add_all_indicators().
    """
    if len(df) < 30:
        return False

    last = df.iloc[-1]

    # 条件1：30日波动幅度 ≥ amp30_min
    if pd.isna(last.get("amp30")) or last["amp30"] < amp30_min:
        return False

    # 条件2：ATR%当前值 > 近90日均值 × ratio
    atr_now = last.get("atr_pct")
    atr_ma = last.get("atr_pct_ma90")
    if pd.isna(atr_now) or pd.isna(atr_ma) or atr_ma <= 0:
        return False
    if atr_now <= atr_ma * atr_pct_ratio:
        return False

    # 条件3：BB带宽 ≥ 近90日 quantile 分位
    if "bb_width" not in df.columns or pd.isna(last["bb_width"]):
        return False
    width_series = df["bb_width"].dropna()
    if len(width_series) < 30:
        return False
    recent = width_series.iloc[-lookback:] if len(width_series) >= lookback else width_series
    threshold = recent.quantile(bb_width_quantile)
    if last["bb_width"] < threshold:
        return False

    return True


def filter_universe(stock_data: dict,
                    amp30_min: float = 0.15,
                    atr_pct_ratio: float = 1.5,
                    bb_width_quantile: float = 0.80,
                    lookback: int = 90) -> list:
    """Returns list of codes that pass volatility gate."""
    passed = []
    for code, df in stock_data.items():
        if passes_volatility(df, amp30_min, atr_pct_ratio, bb_width_quantile, lookback):
            passed.append(code)
    return passed
