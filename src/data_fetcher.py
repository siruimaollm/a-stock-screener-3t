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
            # 仅当本地数据已覆盖到（或超过）end_date 时才跳过
            # 此前的 "5 天缓冲" 会导致周一收盘后 4/24→4/27 这种正常增量被跳过
            if last_dt >= end_dt:
                continue
            # 增量起点 = 上次最后日期的次日，避免重复拉取已有日期
            fetch_from = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            fetch_from = start_date
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


def load_all_kline_sqlite(db_path: str,
                          table: str = "daily_data_hfq",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load kline from stock_data.db (DuckDB format).

    Columns in source: stock_code, date, open, high, low, close,
                       volume, amount, turnover_rate
    Returned columns:  code, date, open, high, low, close,
                       volume, amount, turn, pct_chg, preclose
    """
    # stock_data.db 是 SQLite 格式，用 sqlite3 读取后转 pandas
    import sqlite3 as _sqlite3
    con = _sqlite3.connect(db_path)
    where_parts, params = [], []
    if start_date:
        where_parts.append("date >= ?")
        params.append(start_date)
    if end_date:
        where_parts.append("date <= ?")
        params.append(end_date)
    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    sql = f"SELECT * FROM {table} {where_sql} ORDER BY stock_code, date"
    df = pd.read_sql_query(sql, con, params=params)
    con.close()

    # Rename to match existing pipeline schema
    df = df.rename(columns={"stock_code": "code", "turnover_rate": "turn"})

    # Compute preclose and pct_chg per stock
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["preclose"] = df.groupby("code")["close"].shift(1)
    df["pct_chg"]  = (df["close"] - df["preclose"]) / df["preclose"].replace(0, float("nan"))

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_stock_info_sqlite(db_path: str) -> dict:
    """
    Load stock name + industry from stock_data.db (SQLite format).
    Returns {code: {"name": ..., "industry": ...}}.
    """
    import sqlite3 as _sqlite3
    con = _sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT stock_code, stock_name, industry FROM stock_info"
    ).fetchall()
    con.close()

    result = {}
    for code, name, industry in rows:
        result[code] = {"name": name or "", "industry": industry or ""}
    return result


def load_benchmark_kline(db_path: str,
                         index_code: str = "000300",
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load benchmark index kline from stock_data.db index_data table (SQLite format).
    index_code: '000300' (沪深300), '000001' (上证指数), etc.
    Returns DataFrame with columns: date, close
    """
    import sqlite3 as _sqlite3
    con = _sqlite3.connect(db_path)
    where_parts = ["index_code = ?"]
    params = [index_code]
    if start_date:
        where_parts.append("date >= ?")
        params.append(start_date)
    if end_date:
        where_parts.append("date <= ?")
        params.append(end_date)
    where_sql = "WHERE " + " AND ".join(where_parts)
    sql = f"SELECT date, close FROM index_data {where_sql} ORDER BY date"
    df = pd.read_sql_query(sql, con, params=params)
    con.close()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
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


# ─────────────────────────────────────────────────────────────────────────────
#  stock_data.db 增量更新（SQLite 格式）
# ─────────────────────────────────────────────────────────────────────────────

def _bs_code(stock_code: str) -> str:
    """把 '600519' 转成 BaoStock 格式 'sh.600519'。"""
    if stock_code.startswith(("6", "9")):
        return f"sh.{stock_code}"
    return f"sz.{stock_code}"


def _fetch_batch_sqlite(args):
    """
    子进程：只拉取后复权（adjustflag=1）日线，返回 list of (stock_code, rows_list)。
    rows_list 每项: (stock_code, date, open, high, low, close, volume, amount, turnover_rate)
    """
    batch, retry = args   # batch: list of (stock_code, start_date, end_date)
    import baostock as _bs

    _bs.login()
    results = []
    fields = "date,open,high,low,close,volume,amount,turn"

    for stock_code, start, end in batch:
        bs_code = _bs_code(stock_code)
        rows_out = []
        for attempt in range(retry):
            try:
                rs = _bs.query_history_k_data_plus(
                    bs_code, fields,
                    start_date=start, end_date=end,
                    frequency="d", adjustflag="1",   # 后复权
                )
                if rs.error_code != "0":
                    time.sleep(0.3 * (attempt + 1))
                    continue
                while rs.next():
                    r = rs.get_row_data()
                    # r: [date, open, high, low, close, volume, amount, turn]
                    try:
                        c = float(r[4])
                        if c <= 0:
                            continue
                        rows_out.append((
                            stock_code, r[0],
                            _safe_float(r[1]), _safe_float(r[2]),
                            _safe_float(r[3]), c,
                            _safe_float(r[5]), _safe_float(r[6]),
                            _safe_float(r[7]),
                        ))
                    except (ValueError, IndexError):
                        continue
                break
            except Exception:
                time.sleep(0.5)
        results.append((stock_code, rows_out))

    _bs.logout()
    return results


def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _fetch_index_sqlite(index_bs_code: str, index_code: str,
                        start_date: str, end_date: str, retry: int = 3):
    """拉取指数日线，返回 DataFrame，columns: index_code, date, open, high, low, close, volume, amount。"""
    for attempt in range(retry):
        try:
            rs = bs.query_history_k_data_plus(
                index_bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="3",
            )
            if rs.error_code != "0":
                time.sleep(0.5)
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                             "close", "volume", "amount"])
            df["index_code"] = index_code
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[["index_code", "date", "open", "high", "low", "close", "volume", "amount"]]
        except Exception:
            time.sleep(1)
    return None


def update_stock_data(db_path: str,
                      end_date: Optional[str] = None,
                      workers: int = 8,
                      retry: int = 3) -> dict:
    """
    增量更新 stock_data.db（SQLite 格式）。
    只更新 daily_data_hfq（后复权，分析用）和 index_data。

    优化点：
    - 只拉后复权，API 调用数减半
    - 小批次（100只/批）让多进程负载均衡
    - INSERT OR IGNORE + 唯一索引，无需事后 dedup
    - executemany 批量写入，比 to_sql 快 5-10x

    返回 {"ok": int, "fail": int, "latest_date": str}
    """
    import sqlite3 as _sqlite3
    from datetime import timedelta

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    con = _sqlite3.connect(db_path, timeout=60)
    # WAL 模式：允许读写并发，减少锁竞争
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")

    # ── 1. 找各股最新日期，用全局最小最新日期作为统一起点 ──
    # 增量更新场景下大多数票最新日期相同，直接取最小值统一处理
    row = con.execute("SELECT MIN(mx) FROM (SELECT MAX(date) mx FROM daily_data_hfq GROUP BY stock_code)").fetchone()
    global_min_latest = row[0] if row and row[0] else "2021-01-01"

    # 对最新日期 < 全局最小最新日期的股票（新股等），单独处理
    per_stock_latest = dict(con.execute(
        "SELECT stock_code, MAX(date) FROM daily_data_hfq GROUP BY stock_code"
    ).fetchall())

    all_codes = [r[0] for r in con.execute("SELECT stock_code FROM stock_info").fetchall()]
    con.close()

    from datetime import timedelta
    todo = []
    for code in all_codes:
        last = per_stock_latest.get(code)
        if last:
            last_dt = datetime.strptime(last, "%Y-%m-%d")
            if last_dt >= end_dt:
                continue
            fetch_from = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            fetch_from = "2021-01-01"
        todo.append((code, fetch_from, end_date))

    if not todo:
        print("  stock_data.db: 所有股票已是最新，无需更新。")
        _update_index_data(db_path, end_date, end_dt, retry)
        return {"ok": 0, "fail": 0, "latest_date": end_date}

    print(f"  需更新 {len(todo)} 只股票（{workers} 进程，每批100只）...")

    # ── 2. 小批次，每批 100 只，让所有 worker 均衡负载 ──
    BATCH = 100
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    job_args = [(b, retry) for b in batches]

    ok = fail = 0
    # 由于 fetch 时已按日期过滤（fetch_from = last_date+1），不会有重复行，直接 INSERT
    INSERT_SQL = """
        INSERT INTO daily_data_hfq
        (stock_code, date, open, high, low, close, volume, amount, turnover_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    con = _sqlite3.connect(db_path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-64000")   # 64MB cache

    with mp.Pool(processes=workers) as pool:
        for batch_results in tqdm(
            pool.imap_unordered(_fetch_batch_sqlite, job_args),
            total=len(batches),
            desc="  更新进度",
        ):
            rows_all = []
            for code, rows in batch_results:
                if rows:
                    rows_all.extend(rows)
                    ok += 1
                else:
                    fail += 1
            if rows_all:
                con.executemany(INSERT_SQL, rows_all)
                con.commit()

    con.close()

    # ── 3. 更新指数 ──
    _update_index_data(db_path, end_date, end_dt, retry)

    # ── 4. 统计最终最新日期 ──
    con2 = _sqlite3.connect(db_path)
    latest_date = con2.execute("SELECT MAX(date) FROM daily_data_hfq").fetchone()[0]
    con2.close()

    print(f"  更新完成: OK={ok}  失败/空={fail}  最新日期={latest_date}")
    return {"ok": ok, "fail": fail, "latest_date": latest_date}


def _update_index_data(db_path: str, end_date: str, end_dt, retry: int = 3):
    """更新 index_data 表（沪深300 / 上证 / 深成指）。"""
    import sqlite3 as _sqlite3
    from datetime import timedelta

    index_map = {
        "000300": "sh.000300",
        "000001": "sh.000001",
        "399001": "sz.399001",
    }
    INSERT_IDX = """
        INSERT INTO index_data
        (index_code, date, open, high, low, close, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    con = _sqlite3.connect(db_path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    idx_latest = dict(con.execute(
        "SELECT index_code, MAX(date) FROM index_data GROUP BY index_code"
    ).fetchall())

    for idx_code, bs_code in index_map.items():
        last = idx_latest.get(idx_code)
        if last:
            last_dt = datetime.strptime(last, "%Y-%m-%d")
            if last_dt >= end_dt:
                continue
            idx_start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            idx_start = "2020-01-01"
        df_idx = _fetch_index_sqlite(bs_code, idx_code, idx_start, end_date, retry)
        if df_idx is not None and not df_idx.empty:
            rows = list(df_idx[["index_code", "date", "open", "high", "low",
                                 "close", "volume", "amount"]].itertuples(index=False, name=None))
            con.executemany(INSERT_IDX, rows)
            con.commit()
            print(f"    指数 {idx_code}: +{len(rows)} 行")

    con.close()


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
