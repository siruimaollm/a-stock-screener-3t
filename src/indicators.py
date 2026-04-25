"""
Technical indicator calculations: RSI, KDJ, Bollinger Bands, %B, ATR, 30-day amplitude.
All computed in pure pandas/numpy — no TA-Lib dependency.

KDJ note: domestic brokers use SMA(3) smoothing, which is equivalent to
ewm(alpha=1/3, adjust=False) with initial value 50. Using ewm(span=3) is WRONG
because that gives alpha=2/4.

Defaults follow the reference document "A股大幅波动股票筛选策略 BOLL × RSI × KDJ":
  RSI 双周期: 9 + 14
  KDJ:       9, 3, 3
  BOLL:      20, 2
"""
import numpy as np
import pandas as pd


def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi{period}")


def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 9, smooth: int = 3) -> pd.DataFrame:
    """KDJ with SMA(smooth) via ewm(alpha=1/smooth, adjust=False), initial=50."""
    alpha = 1 / smooth
    low_n = low.rolling(period).min()
    high_n = high.rolling(period).max()
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = (close - low_n) / denom * 100

    k = rsv.copy().astype(float)
    d = rsv.copy().astype(float)

    first_valid = rsv.first_valid_index()
    if first_valid is None:
        return pd.DataFrame({"K": k, "D": d, "J": k})

    loc = rsv.index.get_loc(first_valid)
    k.iloc[loc] = 50 + alpha * (rsv.iloc[loc] - 50)
    d.iloc[loc] = 50 + alpha * (k.iloc[loc] - 50)

    for i in range(loc + 1, len(rsv)):
        k.iloc[i] = k.iloc[i - 1] + alpha * (rsv.iloc[i] - k.iloc[i - 1])
        d.iloc[i] = d.iloc[i - 1] + alpha * (k.iloc[i] - d.iloc[i - 1])

    j = 3 * k - 2 * d
    return pd.DataFrame({"K": k, "D": d, "J": j})


def calc_bollinger(close: pd.Series, period: int = 20,
                   n_std: float = 2) -> pd.DataFrame:
    """
    Bollinger Bands with population std (ddof=0).
    Returns: bb_upper, bb_mid, bb_lower, bb_width, pct_b.
      bb_width = (upper - lower) / mid       (PDF: 带宽 BW)
      pct_b    = (close - lower) / (upper - lower)   (PDF: %B)
    """
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    band_range = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / band_range
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid": mid,
        "bb_lower": lower,
        "bb_width": width,
        "pct_b": pct_b,
    })


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    """ATR via Wilder smoothing; returns atr_pct = ATR/close."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return (atr / close.replace(0, np.nan)).rename("atr_pct")


def calc_amplitude_30(high: pd.Series, low: pd.Series) -> pd.Series:
    """30-day amplitude = (max_high - min_low) / min_low. Captures 30-day price range."""
    max_h = high.rolling(30).max()
    min_l = low.rolling(30).min()
    return ((max_h - min_l) / min_l.replace(0, np.nan)).rename("amp30")


def add_all_indicators(df: pd.DataFrame,
                       rsi_periods=(6, 12, 24),
                       kdj_period=9, kdj_smooth=3,
                       boll_period=14, boll_std=2,
                       atr_period=14) -> pd.DataFrame:
    """
    Accepts a DataFrame with columns [date, open, high, low, close, volume]
    sorted by date ascending. Returns df with all indicator columns appended.

    Default RSI periods include both the user's original (6,12,24) and the
    PDF reference (9,14) for dual-period scoring.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    for p in rsi_periods:
        df[f"rsi{p}"] = calc_rsi(close, p).values

    kdj = calc_kdj(high, low, close, kdj_period, kdj_smooth)
    df["K"] = kdj["K"].values
    df["D"] = kdj["D"].values
    df["J"] = kdj["J"].values

    boll = calc_bollinger(close, boll_period, boll_std)
    for col in boll.columns:
        df[col] = boll[col].values

    df["atr_pct"] = calc_atr(high, low, close, atr_period).values
    df["amp30"] = calc_amplitude_30(high, low).values

    # ATR% rolling 90-day mean — used by volatility filter (PDF: ATR%当前 > 90日均值×1.5)
    df["atr_pct_ma90"] = df["atr_pct"].rolling(90, min_periods=30).mean()

    df["volume"] = df["volume"].astype(float)
    df["vol_ma5"] = df["volume"].rolling(5).mean()

    return df
