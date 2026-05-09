"""
Write pick results to CSV.
"""
import os
import pandas as pd


_COLS = [
    "rank", "code", "name", "industry",
    "total_score", "candle_score", "boll_score", "rsi_score", "kdj_score", "vol_score",
    "confirmation_tier",
    "has_divergence",
    "close", "open", "pct_b",
    "rsi6", "rsi12", "rsi24",
    "K", "D", "J",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "atr_pct", "amp30",
    "vol_ratio", "amount_ma5",
    "signals",
]


def save_csv(picks: list[dict], meta: dict[str, dict],
             output_path: str):
    """
    picks: list of score dicts (from scoring.score_all)
    meta: {code -> {name, industry}} from stock basic
    """
    rows = []
    for rank, p in enumerate(picks, 1):
        code = p["code"]
        m = meta.get(code, {})
        row = {
            "rank": rank,
            "code": code,
            "name": m.get("name", ""),
            "industry": m.get("industry", ""),
            "total_score": p["total_score"],
            "boll_score": p["boll_score"],
            "rsi_score": p["rsi_score"],
            "kdj_score": p["kdj_score"],
            "vol_score": p.get("vol_score", 0),
            "candle_score": p.get("candle_score", 0),
            "confirmation_tier": p.get("confirmation_tier", "none"),
            "has_divergence": p.get("has_divergence", False),
            "close": round(float(p.get("close", 0)), 3),
            "open": round(float(p.get("open", 0) or 0), 3),
            "pct_b": round(float(p.get("pct_b", 0) or 0), 4),
            "rsi6": round(float(p.get("rsi6", 0) or 0), 2),
            "rsi12": round(float(p.get("rsi12", 0) or 0), 2),
            "rsi24": round(float(p.get("rsi24", 0) or 0), 2),
            "K": round(float(p.get("K", 0) or 0), 2),
            "D": round(float(p.get("D", 0) or 0), 2),
            "J": round(float(p.get("J", 0) or 0), 2),
            "bb_upper": round(float(p.get("bb_upper", 0) or 0), 3),
            "bb_mid": round(float(p.get("bb_mid", 0) or 0), 3),
            "bb_lower": round(float(p.get("bb_lower", 0) or 0), 3),
            "bb_width": round(float(p.get("bb_width", 0) or 0), 4),
            "atr_pct": round(float(p.get("atr_pct", 0) or 0), 4),
            "amp30": round(float(p.get("amp30", 0) or 0), 4),
            "vol_ratio": round(float(p.get("vol_ratio", 0) or 0), 2),
            "amount_ma5": round(float(p.get("amount_ma5", 0) or 0), 0),
            "signals": ";".join(p.get("signals", [])),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=_COLS)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"CSV saved: {output_path} ({len(df)} rows)")
    return df
