"""
Build the A-share main board stock universe via BaoStock.

Keeps:
  sh.600xxx / sh.601xxx / sh.603xxx / sh.605xxx
  sz.000xxx / sz.001xxx / sz.002xxx / sz.003xxx

Removes: ST/*ST, 科创板(688), 创业板(300), 北交所(8xx/4xx),
         newly listed < min_listed_days, suspended 3+ days recently.
"""
import re
from datetime import datetime, timedelta
from typing import Optional
import baostock as bs
import pandas as pd


_MAIN_BOARD_PATTERN = re.compile(
    r"^(sh\.(?:600|601|603|605)\d{3}|sz\.(?:000|001|002|003)\d{3})$"
)


def _is_main_board(code: str) -> bool:
    return bool(_MAIN_BOARD_PATTERN.match(code))


def get_universe(min_listed_days: int = 60,
                 exclude_st: bool = True,
                 as_of_date: Optional[str] = None,
                 db_path: Optional[str] = None) -> list:
    """
    Returns list of stock codes eligible for strategy.
    Also caches name/industry to db_path if provided (avoids repeat network calls).
    BaoStock session must already be logged in.
    """
    if as_of_date is None:
        as_of_date = datetime.today().strftime("%Y-%m-%d")

    cutoff = (datetime.strptime(as_of_date, "%Y-%m-%d")
              - timedelta(days=min_listed_days)).strftime("%Y-%m-%d")

    rs = bs.query_stock_basic()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        raise RuntimeError(f"query_stock_basic failed: {rs.error_msg}")

    df = pd.DataFrame(rows, columns=rs.fields)

    # keep main board only
    df = df[df["code"].apply(_is_main_board)].copy()

    # exclude ST / *ST
    if exclude_st:
        df = df[~df["code_name"].str.contains(r"ST|退", na=False)]

    # exclude newly listed
    df = df[df["ipoDate"] <= cutoff]

    # exclude if outDate is set (delisted)
    df = df[df["outDate"].isna() | (df["outDate"] == "")]

    # cache name/industry to DB so picks don't need a second network call
    if db_path:
        from src.data_fetcher import save_stock_basic
        basic_rows = [{"code": r["code"], "code_name": r["code_name"],
                       "industry": r.get("industry", "")}
                      for _, r in df.iterrows()]
        save_stock_basic(db_path, basic_rows)

    return df["code"].tolist()
