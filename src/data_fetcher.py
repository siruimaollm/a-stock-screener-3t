"""
BaoStock data fetcher with DuckDB persistence.

Table schema:
  daily_kline (code TEXT, date TEXT, open DOUBLE, high DOUBLE, low DOUBLE,
               close DOUBLE, preclose DOUBLE, volume DOUBLE, amount DOUBLE,
               turn DOUBLE, pct_chg DOUBLE,
               PRIMARY KEY (code, date))

  ingest_meta (code TEXT PRIMARY KEY, last_date TEXT, row_count INTEGER)
"""
import os
import time
import multiprocessing as mp
from datetime import datetime
from typing import Optional

import baostock as bs
import duckdb
import pandas as pd
from tqdm import tqdm


_FIELDS = "date,open,high,low,close,preclose,volume,amount,turn,pctChg"
_RENAME = {"pctChg": "pct_chg"}
_NUMERIC = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pct_chg"]


def _init_db(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_kline (
            code     TEXT,
            date     TEXT,
            open     DOUBLE,
            high     DOUBLE,
            low      DOUBLE,
            close    DOUBLE,
            preclose DOUBLE,
            volume   DOUBLE,
            amount   DOUBLE,
            turn     DOUBLE,
            pct_chg  DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_meta (
            code       TEXT PRIMARY KEY,
            last_date  TEXT,
            row_count  INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_basic (
            code     TEXT PRIMARY KEY,
            name     TEXT,
            industry TEXT
        )
    """)


def _fetch_one(code: str, start_date: str, end_date: str,
               retry: int = 3) -> Optional[pd.DataFrame]:
    """Fetch daily kline for one stock. Returns None on failure."""
    for attempt in range(retry):
        try:
            rs = bs.query_history_k_data_plus(
                code, _FIELDS,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 前复权
            )
            if rs.error_code != "0":
                time.sleep(0.5 * (attempt + 1))
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=rs.fields)
            df = df.rename(columns=_RENAME)
            df["code"] = code
            for col in _NUMERIC:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            return df[["code", "date"] + _NUMERIC]
        except Exception:
            time.sleep(1)
    return None


def _batch_worker(args):
    """
    Runs in a child process. Each process has its own BaoStock connection.
    args: (batch: list of (code, start, end), retry: int)
    Returns list of (code, df_or_None).
    """
    batch, retry = args
    import baostock as _bs
    _bs.login()
    results = []
    for code, s, e in batch:
        df = None
        for attempt in range(retry):
            try:
                rs = _bs.query_history_k_data_plus(
                    code, _FIELDS,
                    start_date=s, end_date=e,
                    frequency="d", adjustflag="2",
                )
                if rs.error_code != "0":
                    time.sleep(0.3 * (attempt + 1))
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    break
                df = pd.DataFrame(rows, columns=rs.fields)
                df = df.rename(columns=_RENAME)
                df["code"] = code
                for col in _NUMERIC:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                df = df[df["close"] > 0]
                df = df[["code", "date"] + _NUMERIC]
                break
            except Exception:
                time.sleep(0.5)
        results.append((code, df))
    _bs.logout()
    return results


def ingest(codes: list, db_path: str,
           start_date: str = "2022-04-01",
           end_date: Optional[str] = None,
           workers: int = 8,
           retry: int = 3):
    """
    Bulk-ingest daily kline for all codes into DuckDB.
    Uses multiprocessing so each worker has its own BaoStock connection.
    Incremental: skips codes already up-to-date.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    conn = duckdb.connect(db_path)
    _init_db(conn)

    meta = conn.execute("SELECT code, last_date FROM ingest_meta").df()
    meta_map = dict(zip(meta["code"], meta["last_date"]))

    from datetime import timedelta
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    todo = []
    for code in codes:
        last = meta_map.get(code)
        if last:
            last_dt = datetime.strptime(last, "%Y-%m-%d")
            # skip if data is within 5 calendar days of end_date
            # (handles weekends/holidays where BaoStock hasn't published today yet)
            if (end_dt - last_dt).days <= 5:
                continue
        fetch_from = last if last else start_date
        todo.append((code, fetch_from, end_date))

    if not todo:
        print("All codes already up-to-date.")
        conn.close()
        return

    print(f"Fetching {len(todo)} stocks with {workers} processes ...")

    # split into equal batches, one per process
    batch_size = max(1, len(todo) // workers)
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    job_args = [(b, retry) for b in batches]

    ok, fail = 0, 0
    with mp.Pool(processes=workers) as pool:
        for batch_results in tqdm(
            pool.imap_unordered(_batch_worker, job_args),
            total=len(batches),
            desc="Batches",
        ):
            for code, df in batch_results:
                if df is None or df.empty:
                    fail += 1
                    continue
                conn.execute("INSERT OR REPLACE INTO daily_kline SELECT * FROM df")
                conn.execute(
                    "INSERT OR REPLACE INTO ingest_meta VALUES (?, ?, ?)",
                    [code, df["date"].max(), len(df)],
                )
                ok += 1
            conn.commit()

    conn.close()
    print(f"Done. OK={ok}  Failed={fail}")


def load_kline(code: str, db_path: str,
               start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> pd.DataFrame:
    """Load kline from DuckDB for one stock, sorted by date asc."""
    conn = duckdb.connect(db_path, read_only=True)
    where = ["code = ?"]
    params = [code]
    if start_date:
        where.append("date >= ?")
        params.append(start_date)
    if end_date:
        where.append("date <= ?")
        params.append(end_date)
    q = f"SELECT * FROM daily_kline WHERE {' AND '.join(where)} ORDER BY date"
    df = conn.execute(q, params).df()
    conn.close()
    return df


def load_all_kline(db_path: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """Load all stocks kline from DuckDB."""
    conn = duckdb.connect(db_path, read_only=True)
    where = []
    params = []
    if start_date:
        where.append("date >= ?")
        params.append(start_date)
    if end_date:
        where.append("date <= ?")
        params.append(end_date)
    q = "SELECT * FROM daily_kline"
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY code, date"
    df = conn.execute(q, params).df()
    conn.close()
    return df


def get_db_stats(db_path: str) -> dict:
    """Return basic stats about the database."""
    conn = duckdb.connect(db_path, read_only=True)
    total = conn.execute("SELECT COUNT(*) FROM daily_kline").fetchone()[0]
    stocks = conn.execute("SELECT COUNT(DISTINCT code) FROM daily_kline").fetchone()[0]
    min_d = conn.execute("SELECT MIN(date) FROM daily_kline").fetchone()[0]
    max_d = conn.execute("SELECT MAX(date) FROM daily_kline").fetchone()[0]
    conn.close()
    return {"total_rows": total, "stocks": stocks, "min_date": min_d, "max_date": max_d}


def save_stock_basic(db_path: str, rows: list):
    """Cache stock name/industry in DB to avoid repeated query_stock_basic calls."""
    conn = duckdb.connect(db_path)
    _init_db(conn)
    for r in rows:
        conn.execute(
            "INSERT OR REPLACE INTO stock_basic VALUES (?, ?, ?)",
            [r.get("code", ""), r.get("code_name", ""), r.get("industry", "")],
        )
    conn.commit()
    conn.close()


def load_stock_basic(db_path: str) -> dict:
    """Load cached stock name/industry from DB. Returns {code: {name, industry}}."""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        df = conn.execute("SELECT code, name, industry FROM stock_basic").df()
        conn.close()
        return {r["code"]: {"name": r["name"], "industry": r["industry"]}
                for _, r in df.iterrows()}
    except Exception:
        return {}
