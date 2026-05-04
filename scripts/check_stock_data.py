import duckdb, sqlite3, os

db = "data/stock_data.db"

# Try DuckDB
try:
    conn = duckdb.connect(db, read_only=True)
    tables = conn.execute("SHOW TABLES").fetchall()
    print("Format: DuckDB")
    print("Tables:", [t[0] for t in tables])
    for t in tables:
        cnt = conn.execute("SELECT COUNT(*) FROM " + t[0]).fetchone()[0]
        cols = [c[0] for c in conn.execute("DESCRIBE " + t[0]).fetchall()]
        sample = conn.execute("SELECT * FROM " + t[0] + " LIMIT 2").fetchall()
        print(f"  {t[0]}: {cnt} rows | cols: {cols}")
        print(f"    sample: {sample}")
    conn.close()
except Exception as e:
    print("DuckDB failed:", e)
    try:
        conn = sqlite3.connect(db)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print("Format: SQLite")
        print("Tables:", [t[0] for t in tables])
        for t in tables:
            cnt = conn.execute("SELECT COUNT(*) FROM " + t[0]).fetchone()[0]
            cols = [c[1] for c in conn.execute("PRAGMA table_info(" + t[0] + ")").fetchall()]
            sample = conn.execute("SELECT * FROM " + t[0] + " LIMIT 2").fetchall()
            print(f"  {t[0]}: {cnt} rows | cols: {cols}")
            print(f"    sample: {sample}")
        conn.close()
    except Exception as e2:
        print("SQLite failed:", e2)
