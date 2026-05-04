"""
Layer 2 — 五维共振评分体系（重构版，满分 15 分）

策略定位：筛出「曾剧烈波动、近期BB带宽收窄整理、触及下轨后反转」的超卖反弹候选。

============================================================
维度1 BOLL（0-3分）— 必要条件维度
  核心要求：近3日触及布林下轨 + 带宽收窄
  3分: 近3日触及下轨 + 带宽收窄 + %B今日回升（开始反弹）
  2分: 近3日触及下轨 + 带宽收窄
  1分: 近3日触及下轨（带宽未明显收窄）
  0分: 未触及下轨 → 此维度为 0 时整只股票强制淘汰

============================================================
维度2 RSI（0-3分）
  RSI(6) < 40 为基本要求，底背离为加分项
  3分: RSI6 < 40 + 底背离 + RSI6今日>昨日（开始回升）
  2分: RSI6 < 40 + 底背离
  1分: RSI6 < 40（无底背离）
  0分: RSI6 ≥ 40

  底背离定义：价格创20日新低 + RSI6未创新低 + RSI6 < 40

============================================================
维度3 KDJ（0-3分）
  核心要求：K上穿D（金叉）+ J带头向上
  3分: K上穿D + J向上 + K<30（低位金叉）
  2分: K上穿D + J向上
  1分: K上穿D（J未必向上）或 J向上且K<30
  0分: 无金叉且无J向上低位信号

============================================================
维度4 量价（0-3分）— 多路径明显止跌确认
  3分路径A: 连续3日量比递减+均值<0.75 + 收盘≥当日(H+L)/2 + 5日未创收盘新低
  3分路径B: 前3日缩量（均值<0.7）后今日放量阳线（承接确认）
  2分: 今日量比<0.65 + 收盘≥当日(H+L)/2（单日明显缩量且收盘强势）
  1分: 5日均量比<0.85 + 收盘≥昨收×0.995（温和缩量价稳）
  0分: 放量阴线 或 不满足以上任何路径

============================================================
维度5 蜡烛形态（0-3分）
  核心要求：金针探底（长下影+小实体+收盘高位）
  3分: 创近10日新低 + 下影线≥实体×3 + 收盘位于K线≥70% + 实体≤30%
  2分: 下影线≥实体×2 + 收盘位于K线≥65% + 实体≤40%
  1分: 收盘位于K线≥60% + 收阳线
  0分: 大阴线/长上影/无下影

============================================================
入选标准（总分满分15）：
  ① BOLL维度 ≥ 1（必须触及下轨，不满足直接淘汰）
  ② 量价止跌确认（三档任一）：
       vol≥2  OR  candle≥3  OR  (vol≥1 AND candle≥2)
  ③ total_score ≥ 9
  排序：total_score DESC，candle_score DESC，rsi底背离优先
"""
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  辅助函数
# ─────────────────────────────────────────────

def detect_bullish_divergence(close: pd.Series, rsi: pd.Series,
                               lookback: int = 20) -> bool:
    """
    看涨底背离：
      1) 当前价格创近 lookback 日新低
      2) 当前 RSI6 高于窗口内最低 RSI6（RSI未创新低）
      3) 当前 RSI6 < 40
    """
    if len(close) < lookback + 1 or len(rsi) < lookback + 1:
        return False
    curr_close = float(close.iloc[-1])
    curr_rsi = float(rsi.iloc[-1])
    if pd.isna(curr_close) or pd.isna(curr_rsi) or curr_rsi >= 40:
        return False
    past_close = close.iloc[-(lookback + 1):-1]
    past_rsi = rsi.iloc[-(lookback + 1):-1]
    if past_close.empty or past_rsi.dropna().empty:
        return False
    if curr_close >= float(past_close.min()):
        return False
    if curr_rsi <= float(past_rsi.min()):
        return False
    return True


def _bw_narrowing(df: pd.DataFrame) -> bool:
    """当前BB带宽 < 近20日峰值×0.75（带宽收窄中）。"""
    if "bb_width" not in df.columns or len(df) < 20:
        return False
    bw = df["bb_width"].dropna()
    if len(bw) < 10:
        return False
    bw_now = float(bw.iloc[-1])
    bw_max20 = float(bw.iloc[-20:].max())
    return bw_max20 > 0 and bw_now < bw_max20 * 0.75


# ─────────────────────────────────────────────
#  五个评分函数
# ─────────────────────────────────────────────

def score_boll(df: pd.DataFrame) -> tuple:
    """
    BOLL评分（0-3分）。
    必要条件：近3日有 low ≤ bb_lower。
    3分: 带宽收窄 + 连续2日%B回升 + %B<0.3（明确从下轨反弹）
    2分: 带宽收窄 + 单日%B回升
    1分: 触及下轨（带宽未明显收窄）
    """
    if len(df) < 10 or "bb_lower" not in df.columns:
        return 0, []

    # 近3日是否触及下轨
    recent3 = df.iloc[-3:]
    touched = bool((recent3["low"].astype(float) <= recent3["bb_lower"].astype(float)).any())
    if not touched:
        return 0, []

    narrowing = _bw_narrowing(df)

    pct_b = float(df["pct_b"].iloc[-1]) if "pct_b" in df.columns else np.nan

    # %B连续2日回升（更严格确认）
    pct_b_rebounding_2d = False
    pct_b_rebounding_1d = False
    if "pct_b" in df.columns and len(df) >= 3:
        pb_now   = df["pct_b"].iloc[-1]
        pb_prev  = df["pct_b"].iloc[-2]
        pb_prev2 = df["pct_b"].iloc[-3]
        if not any(pd.isna(x) for x in [pb_now, pb_prev, pb_prev2]):
            pb_now, pb_prev, pb_prev2 = float(pb_now), float(pb_prev), float(pb_prev2)
            pct_b_rebounding_2d = (pb_now > pb_prev > pb_prev2) and pb_now < 0.3
            pct_b_rebounding_1d = pb_now > pb_prev
    elif "pct_b" in df.columns and len(df) >= 2:
        pb_now  = df["pct_b"].iloc[-1]
        pb_prev = df["pct_b"].iloc[-2]
        if not any(pd.isna(x) for x in [pb_now, pb_prev]):
            pct_b_rebounding_1d = float(pb_now) > float(pb_prev)

    if narrowing and pct_b_rebounding_2d:
        return 3, [f"布林下轨触及+带宽收窄+连续%B回升({pct_b:.2f})"]
    if narrowing and pct_b_rebounding_1d:
        return 2, [f"布林下轨触及+带宽收窄+%B回升({pct_b:.2f})"]
    if narrowing:
        return 2, [f"布林下轨触及+带宽收窄({pct_b:.2f})"]
    return 1, [f"布林下轨触及(%B={pct_b:.2f})"]


def score_rsi(df: pd.DataFrame, rsi_oversold: float = 40) -> tuple:
    """
    RSI评分（0-3分）。
    RSI6 < rsi_oversold 为基本要求。
    3分: 底背离 + 连续2日RSI6回升（确认反转）
    2分: 底背离 + 单日回升，或底背离本身
    1分: RSI6<超卖线（无背离）
    """
    if "rsi6" not in df.columns or len(df) < 3:
        return 0, [], False

    rsi6_now   = float(df["rsi6"].iloc[-1])
    rsi6_prev  = float(df["rsi6"].iloc[-2])
    rsi6_prev2 = float(df["rsi6"].iloc[-3])
    if pd.isna(rsi6_now):
        return 0, [], False

    has_div = detect_bullish_divergence(df["close"].astype(float), df["rsi6"].astype(float))

    # 连续2日回升 vs 单日回升
    rising_2d = rsi6_now > rsi6_prev and rsi6_prev > rsi6_prev2
    rising_1d = rsi6_now > rsi6_prev

    if rsi6_now >= rsi_oversold:
        return 0, [], has_div

    if has_div and rising_2d:
        return 3, [f"RSI6底背离+连续回升(RSI6={rsi6_now:.1f})"], True
    if has_div and rising_1d:
        return 2, [f"RSI6底背离+回升(RSI6={rsi6_now:.1f})"], True
    if has_div:
        return 2, [f"RSI6底背离(RSI6={rsi6_now:.1f})"], True
    if rising_1d:
        return 1, [f"RSI6<{rsi_oversold:.0f}+回升(RSI6={rsi6_now:.1f})"], False
    return 1, [f"RSI6<{rsi_oversold:.0f}(RSI6={rsi6_now:.1f})"], False


def score_kdj(df: pd.DataFrame) -> tuple:
    """
    KDJ评分（0-3分）。
    核心信号：K上穿D（金叉）+ J带头向上。
    """
    if not all(c in df.columns for c in ["K", "D", "J"]) or len(df) < 2:
        return 0, []

    k_now  = float(df["K"].iloc[-1]);  k_prev = float(df["K"].iloc[-2])
    d_now  = float(df["D"].iloc[-1]);  d_prev = float(df["D"].iloc[-2])
    j_now  = float(df["J"].iloc[-1]);  j_prev = float(df["J"].iloc[-2])
    if any(pd.isna(x) for x in [k_now, d_now, j_now, k_prev, d_prev, j_prev]):
        return 0, []

    golden_cross = k_prev <= d_prev and k_now > d_now   # K上穿D
    j_rising = j_now > j_prev                             # J向上
    low_pos = k_now < 30 and d_now < 30                  # 低位

    if golden_cross and j_rising and low_pos:
        return 3, [f"KDJ低位金叉+J向上(K={k_now:.0f},D={d_now:.0f},J={j_now:.0f})"]
    if golden_cross and j_rising:
        return 2, [f"KDJ金叉+J向上(K={k_now:.0f},D={d_now:.0f},J={j_now:.0f})"]
    if golden_cross:
        return 1, [f"KDJ金叉(K={k_now:.0f},D={d_now:.0f})"]
    if j_rising and k_now < 30:
        return 1, [f"J向上低位(K={k_now:.0f},J={j_now:.0f})"]
    return 0, []


def score_volume(df: pd.DataFrame) -> tuple:
    """
    量价评分（0-3分）。
    核心信号：量价出现明显止跌特征，多路径判定：

    Path A（3分）: 连续3日量比严格递减 + 3日均值<0.75 + 收盘≥当日中轴(H+L)/2 + 5日未创收盘新低
    Path B（3分）: 前3日量比均值<0.7后今日放量阳线（缩量整理后承接确认）
    Path C（2分）: 今日量比<0.65 + 收盘≥当日中轴（单日明显缩量+收盘强势）
    Path D（1分）: 5日均量比<0.85 + 收盘≥昨收×0.995（温和缩量价稳）
    """
    if "vol_ratio" not in df.columns or len(df) < 6:
        return 0, []

    last = df.iloc[-1]
    vr_now = last.get("vol_ratio", np.nan)
    if pd.isna(vr_now):
        return 0, []
    vr_now = float(vr_now)

    close_now  = float(last["close"]) if not pd.isna(last.get("close")) else None
    close_prev = float(df["close"].iloc[-2]) if len(df) >= 2 else None
    high_now   = float(last["high"])  if not pd.isna(last.get("high"))  else None
    low_now    = float(last["low"])   if not pd.isna(last.get("low"))   else None
    if close_now is None or close_prev is None or high_now is None or low_now is None:
        return 0, []

    # 价格止跌的严格定义：收盘在当日(H+L)/2以上（不只是未大跌）
    mid_price       = (high_now + low_now) / 2
    close_above_mid = close_now >= mid_price

    # 5日内未创收盘新低（排除持续阴跌中的假缩量）
    past5_close_min = float(df["close"].iloc[-6:-1].min()) if len(df) >= 6 else close_now
    price_no_new_low = close_now >= past5_close_min

    # 放量阴线直接归零（出货信号）
    if vr_now >= 1.5 and close_now < close_prev:
        return 0, []

    # ── Path A：连续3日量比递减 + 持续缩量止跌 ──
    vr_series = df["vol_ratio"].iloc[-3:].tolist()
    vr_declining = (len(vr_series) == 3
                    and not any(pd.isna(v) for v in vr_series)
                    and vr_series[0] > vr_series[1] > vr_series[2])
    vr_3d_mean = float(df["vol_ratio"].iloc[-3:].mean())

    if vr_declining and vr_3d_mean < 0.75 and close_above_mid and price_no_new_low:
        return 3, [f"持续缩量止跌(3日量比递减{vr_series[0]:.2f}→{vr_series[2]:.2f},均={vr_3d_mean:.2f},收≥中轴)"]

    # ── Path B：缩量整理后放量阳线确认 ──
    vr_prev3_mean = float(df["vol_ratio"].iloc[-4:-1].mean()) if len(df) >= 4 else np.nan
    if (not pd.isna(vr_prev3_mean)
            and vr_prev3_mean < 0.7
            and vr_now >= 1.1
            and close_now > close_prev):
        return 3, [f"缩量整理后放量阳线(前3日均={vr_prev3_mean:.2f}→今={vr_now:.2f}↑,收阳)"]

    # ── Path C：单日明显缩量+收盘强势 ──
    if vr_now < 0.65 and close_above_mid:
        return 2, [f"单日明显缩量止跌(量比={vr_now:.2f},收{close_now:.2f}≥中轴{mid_price:.2f})"]

    # ── Path D：温和缩量价稳 ──
    vr_5d_mean = float(df["vol_ratio"].iloc[-5:].mean())
    if vr_5d_mean < 0.85 and close_now >= close_prev * 0.995:
        return 1, [f"量缩价稳(5日均量比={vr_5d_mean:.2f})"]

    return 0, []


def score_candle(df: pd.DataFrame) -> tuple:
    """
    蜡烛形态评分（0-3分）。
    核心信号：金针探底（长下影+小实体+收盘高位，在低位创新低后强势收回）。
    """
    if len(df) < 11:
        return 0, []

    last = df.iloc[-1]
    o = last.get("open"); h = last.get("high")
    l = last.get("low");  c = last.get("close")
    if any(pd.isna(x) for x in [o, h, l, c]):
        return 0, []
    o, h, l, c = float(o), float(h), float(l), float(c)

    rng = h - l
    if rng <= 0:
        return 0, []

    body = abs(c - o)
    lower_shadow = min(o, c) - l
    close_pos = (c - l) / rng          # 收盘在K线中的相对位置（0=最低，1=最高）
    body_ratio = body / rng             # 实体占K线比例

    prev10_low = float(df["low"].iloc[-11:-1].min())
    is_new_low10 = l < prev10_low

    # 3分：金针探底（最高信号）
    if (is_new_low10
            and lower_shadow >= body * 3
            and close_pos >= 0.70
            and body_ratio <= 0.30):
        ratio = lower_shadow / (body + 1e-9)
        return 3, [f"金针探底(创10日新低+下影/实体={ratio:.1f}+收盘{close_pos:.0%})"]

    # 2分：普通锤子线
    if (lower_shadow >= body * 2
            and close_pos >= 0.65
            and body_ratio <= 0.40):
        ratio = lower_shadow / (body + 1e-9)
        return 2, [f"锤子线(下影/实体={ratio:.1f}+收盘{close_pos:.0%})"]

    # 1分：收盘强势阳线
    if close_pos >= 0.60 and c >= o:
        return 1, [f"收盘强势阳线({close_pos:.0%})"]

    return 0, []


# ─────────────────────────────────────────────
#  汇总评分
# ─────────────────────────────────────────────

def score_stock(df: pd.DataFrame, rsi_long_oversold: float = 40) -> dict:
    """对单只股票打分（5维：BOLL+RSI+KDJ+量价+蜡烛，满分15）。"""
    result = {
        "boll_score": 0, "rsi_score": 0, "kdj_score": 0,
        "vol_score": 0, "candle_score": 0, "total_score": 0,
        "has_divergence": False, "signals": [],
        "rsi6": np.nan, "rsi12": np.nan, "rsi24": np.nan,
        "K": np.nan, "D": np.nan, "J": np.nan,
        "bb_upper": np.nan, "bb_mid": np.nan, "bb_lower": np.nan,
        "bb_width": np.nan, "pct_b": np.nan,
        "atr_pct": np.nan, "amp30": np.nan,
        "close": np.nan, "open": np.nan, "volume": np.nan,
        "vol_ratio": np.nan, "amount_ma5": np.nan,
    }

    if df is None or len(df) < 30:
        return result

    last = df.iloc[-1]
    for col in ["rsi6", "rsi12", "rsi24", "K", "D", "J",
                "bb_upper", "bb_mid", "bb_lower", "bb_width", "pct_b",
                "atr_pct", "amp30", "close", "open", "volume",
                "vol_ratio", "amount_ma5"]:
        if col in df.columns:
            result[col] = last[col]

    boll,   sig_b         = score_boll(df)
    rsi,    sig_r, has_div = score_rsi(df, rsi_oversold=rsi_long_oversold)
    kdj,    sig_k         = score_kdj(df)
    vol,    sig_v         = score_volume(df)
    candle, sig_c         = score_candle(df)

    result["boll_score"]   = boll
    result["rsi_score"]    = rsi
    result["kdj_score"]    = kdj
    result["vol_score"]    = vol
    result["candle_score"] = candle
    result["total_score"]  = boll + rsi + kdj + vol + candle
    result["has_divergence"] = has_div
    result["signals"] = sig_b + sig_r + sig_k + sig_v + sig_c
    return result


def is_selected(scores: dict, threshold: int = 9,
                require_all_dimensions: bool = True,
                require_rsi_divergence: bool = False) -> bool:
    """
    入选标准（5维满分15）：
      ① boll_score ≥ 1（必须触及下轨，硬性必要条件）
      ② 量价止跌确认（三档任一达标）：
           vol_score ≥ 2  ← 明显止跌（持续缩量/缩量后放量阳线）
         OR candle_score ≥ 3  ← 极强金针探底
         OR (vol_score ≥ 1 AND candle_score ≥ 2)  ← 温和缩量+锤子线组合
      ③ total_score ≥ threshold（默认9）
      require_rsi_divergence=True 时额外要求底背离（默认不强制）
    """
    # ① BOLL硬性条件
    if scores.get("boll_score", 0) < 1:
        return False

    # ② 量价止跌确认（三档达标）
    vol    = scores.get("vol_score", 0)
    candle = scores.get("candle_score", 0)
    vol_candle_ok = (vol >= 2) or (candle >= 3) or (vol >= 1 and candle >= 2)
    if not vol_candle_ok:
        return False

    # ③ 总分门槛
    if scores["total_score"] < threshold:
        return False

    # 可选：底背离
    if require_rsi_divergence and not scores.get("has_divergence", False):
        return False

    return True


def score_all(stock_data: dict, threshold: int = 10,
              require_all_dimensions: bool = True,
              require_rsi_divergence: bool = False,
              rsi_long_oversold: float = 40, **kwargs) -> list:
    selected = []
    for code, df in stock_data.items():
        s = score_stock(df, rsi_long_oversold=rsi_long_oversold)
        s["code"] = code
        if is_selected(s, threshold, require_all_dimensions, require_rsi_divergence):
            selected.append(s)

    # 排序：总分 > 底背离 > 蜡烛形态 > KDJ
    selected.sort(key=lambda x: (
        x["total_score"],
        1 if x.get("has_divergence") else 0,
        x.get("candle_score", 0),
        x.get("kdj_score", 0),
    ), reverse=True)
    return selected
