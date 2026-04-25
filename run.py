"""
Main CLI entry point.

Usage:
  python run.py validate [--code sh.600519]
  python run.py ingest [--since 2022-04-01]
  python run.py backtest [--auto-tune]
  python run.py pick [--date YYYYMMDD] [--top N]
  python run.py          (runs all four steps in sequence)
"""
import argparse
import os
import sys
from datetime import datetime, date
from typing import Optional

import baostock as bs
import pandas as pd
import yaml


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _last_trading_date() -> str:
    today = date.today()
    # return today's date; BaoStock will handle non-trading-day gaps
    return today.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
#  VALIDATE
# ─────────────────────────────────────────────
def cmd_validate(cfg: dict, code: str = "sh.600519"):
    from src.data_fetcher import _fetch_one, _init_db, ingest, load_kline
    from src.indicators import add_all_indicators
    from src.report_html import save_validate_html
    from src.scoring import score_stock

    print(f"\n=== STEP 1: Validate {code} ===")
    db_path = cfg["data"]["db_path"]
    since = cfg["data"]["ingest_since"]
    today = datetime.today().strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    print("Logging in to BaoStock ...")
    lg = bs.login()
    if lg.error_code != "0":
        print(f"BaoStock login failed: {lg.error_msg}")
        sys.exit(1)

    print(f"Fetching {code} kline from {since} to {today} ...")
    ingest([code], db_path, start_date=since, end_date=today,
           workers=1, retry=cfg["data"]["retry"])

    bs.logout()

    df = load_kline(code, db_path)
    if df.empty:
        print(f"ERROR: No data for {code}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows for {code}.")
    df = add_all_indicators(
        df,
        rsi_periods=cfg["indicators"]["rsi_periods"],
        kdj_period=cfg["indicators"]["kdj_period"],
        kdj_smooth=cfg["indicators"]["kdj_smooth"],
        boll_period=cfg["indicators"]["boll_period"],
        boll_std=cfg["indicators"]["boll_std"],
        atr_period=cfg["indicators"]["atr_period"],
    )

    scores = score_stock(df)
    print(f"\nLatest scoring for {code}:")
    for k in ["rsi_score","kdj_score","boll_score","vol_score","total_score"]:
        print(f"  {k}: {scores[k]}")
    print(f"  signals: {';'.join(scores['signals'])}")

    out_dir = cfg["pick"]["output_dir"]
    out_path = os.path.join(out_dir, f"validate_{code.replace('.','')}.html")
    save_validate_html(df, code, out_path)
    print(f"\nOpen in browser: {os.path.abspath(out_path)}")
    print("Please compare RSI6/J/BB values with Tongdaxin or Tonghuashun (tolerance < 0.5%).")


# ─────────────────────────────────────────────
#  INGEST
# ─────────────────────────────────────────────
def cmd_ingest(cfg: dict, since: Optional[str] = None):
    from src.universe import get_universe
    from src.data_fetcher import ingest, get_db_stats

    print("\n=== STEP 2: Ingest full main board universe ===")
    db_path = cfg["data"]["db_path"]
    start = since or cfg["data"]["ingest_since"]
    today = datetime.today().strftime("%Y-%m-%d")

    print("Logging in to BaoStock ...")
    lg = bs.login()
    if lg.error_code != "0":
        print(f"BaoStock login failed: {lg.error_msg}")
        sys.exit(1)

    print("Building universe ...")
    codes = get_universe(
        min_listed_days=cfg["universe"]["min_listed_days"],
        exclude_st=cfg["universe"]["exclude_st"],
        db_path=db_path,
    )
    print(f"Universe size: {len(codes)} stocks")

    ingest(codes, db_path, start_date=start, end_date=today,
           workers=cfg["data"]["baostock_workers"],
           retry=cfg["data"]["retry"])

    bs.logout()

    stats = get_db_stats(db_path)
    print(f"\nDB stats: {stats}")


# ─────────────────────────────────────────────
#  BACKTEST
# ─────────────────────────────────────────────
def cmd_backtest(cfg: dict, auto_tune: bool = False):
    import duckdb
    from src.backtest import run_backtest, check_pass_criteria
    from src.backtest_report import save_backtest_html

    print("\n=== STEP 3: Backtest ===")
    db_path = cfg["data"]["db_path"]
    bt_cfg = cfg["backtest"]
    pass_criteria = bt_cfg["pass_criteria"]
    hold_periods = bt_cfg["hold_periods"]

    # Load all kline
    conn = duckdb.connect(db_path, read_only=True)
    print("Loading main board kline ...")
    all_kline = conn.execute(
        "SELECT * FROM daily_kline WHERE date >= ? AND date <= ? ORDER BY code, date",
        [bt_cfg["start_date"], bt_cfg["end_date"]]
    ).df()

    bench_code = bt_cfg["benchmark"]
    print(f"Loading benchmark {bench_code} ...")
    try:
        bench_kline = conn.execute(
            "SELECT date, close FROM daily_kline WHERE code = ? ORDER BY date",
            [bench_code]
        ).df()
    except Exception:
        bench_kline = _fetch_benchmark(bench_code, bt_cfg["start_date"], bt_cfg["end_date"])

    conn.close()

    vol_cfg = {"amp30_min": cfg["volatility"]["amp30_min"],
               "atr_pct_ratio": cfg["volatility"]["atr_pct_ratio"],
               "bb_width_quantile": cfg["volatility"]["bb_width_quantile"],
               "lookback": cfg["volatility"].get("lookback", 90)}
    score_cfg = {"threshold": cfg["scoring"]["threshold"],
                 "require_all_dimensions": cfg["scoring"].get("require_all_dimensions", True),
                 "require_rsi_divergence": cfg["scoring"].get("require_rsi_divergence", True),
                 "rsi_long_oversold": cfg["scoring"].get("rsi_long_oversold", 30)}

    results = run_backtest(all_kline, bench_kline,
                           bt_cfg["start_date"], bt_cfg["end_date"],
                           hold_periods, bt_cfg["top_n"],
                           vol_cfg, score_cfg)

    passed, failed = check_pass_criteria(results, pass_criteria)
    auto_tuned = False

    if not passed and auto_tune:
        print(f"\n*** Backtest failed: {failed} ***")
        print("Starting auto-tune grid search ...")
        from src.tuner import grid_search

        conn2 = duckdb.connect(db_path, read_only=True)
        oos_kline = conn2.execute(
            "SELECT * FROM daily_kline WHERE date >= ? AND date <= ? ORDER BY code, date",
            [bt_cfg["out_of_sample_start"], bt_cfg["out_of_sample_end"]]
        ).df()
        conn2.close()

        tune_result = grid_search(
            all_kline, bench_kline, pass_criteria,
            bt_cfg, cfg["tuner"],
            oos_kline=oos_kline,
        )

        if tune_result["best_config"] and tune_result["oos_passed"]:
            print(f"\nBest config found: {tune_result['best_config']}")
            # Update config with tuned params
            best = tune_result["best_config"]
            cfg["volatility"]["amp30_min"] = best["amp30"]
            cfg["scoring"]["threshold"] = best["threshold"]
            if "atr_pct_ratio" in best:
                cfg["volatility"]["atr_pct_ratio"] = best["atr_pct_ratio"]
            if "bb_width_quantile" in best:
                cfg["volatility"]["bb_width_quantile"] = best["bb_width_quantile"]
            # Save tuned config
            with open("config_tuned.yaml", "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True)
            print("Tuned config saved to config_tuned.yaml")
            results = tune_result["best_results"]
            passed, failed = check_pass_criteria(results, pass_criteria)
            auto_tuned = True
        else:
            print("\n*** Auto-tune could not find a passing config. ***")
            print("Strategy does not show significant positive return in this sample.")
            passed = False

    out_dir = cfg["pick"]["output_dir"]
    date_str = datetime.today().strftime("%Y%m%d")
    out_path = os.path.join(out_dir, f"backtest_{date_str}.html")
    save_backtest_html(results, hold_periods, failed, passed, auto_tuned,
                       out_path, bt_cfg["start_date"], bt_cfg["end_date"])

    for h in hold_periods:
        m = results.get(f"metrics_{h}", {})
        print(f"\n--- {h}日持有 ---")
        print(f"  笔数={m.get('count',0)}  胜率={m.get('win_rate',0):.1%}  "
              f"均收益={m.get('mean_return',0):.2%}  超额胜率={m.get('exc_win_rate',0) or 0:.1%}")

    if passed:
        print("\n[PASS] 策略通过验证。")
    else:
        print(f"\n[FAIL] 策略未通过验证: {failed}")

    return passed, cfg


def _fetch_benchmark(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch benchmark index kline directly via BaoStock."""
    import pandas as pd
    lg = bs.login()
    rs = bs.query_history_k_data_plus(code, "date,close",
                                       start_date=start, end_date=end,
                                       frequency="d", adjustflag="3")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


# ─────────────────────────────────────────────
#  PICK
# ─────────────────────────────────────────────
def cmd_pick(cfg: dict, pick_date: Optional[str] = None, top_n: Optional[int] = None):
    import pandas as pd
    import duckdb
    from src.indicators import add_all_indicators
    from src.volatility_filter import passes_volatility
    from src.scoring import score_all
    from src.report_csv import save_csv
    from src.report_html import save_picks_html

    print("\n=== STEP 4: Pick stocks ===")

    if pick_date is None:
        pick_date = _last_trading_date()
    n = top_n or cfg["pick"]["top_n"]
    db_path = cfg["data"]["db_path"]

    # need ~120 bars before pick_date
    from datetime import datetime as dt, timedelta
    since_dt = dt.strptime(pick_date, "%Y-%m-%d") - timedelta(days=200)
    since = since_dt.strftime("%Y-%m-%d")

    conn = duckdb.connect(db_path, read_only=True)
    all_kline = conn.execute(
        "SELECT * FROM daily_kline WHERE date >= ? AND date <= ? ORDER BY code, date",
        [since, pick_date]
    ).df()
    conn.close()

    if all_kline.empty:
        print("No data in DB. Run ingest first.")
        return

    vol_cfg = {"amp30_min": cfg["volatility"]["amp30_min"],
               "atr_pct_ratio": cfg["volatility"]["atr_pct_ratio"],
               "bb_width_quantile": cfg["volatility"]["bb_width_quantile"],
               "lookback": cfg["volatility"].get("lookback", 90)}
    score_cfg = cfg["scoring"]

    print(f"Computing indicators for {all_kline['code'].nunique()} stocks ...")
    stock_data = {}
    for code, grp in all_kline.groupby("code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < 60:
            continue
        stock_data[code] = add_all_indicators(
            grp,
            rsi_periods=cfg["indicators"]["rsi_periods"],
            kdj_period=cfg["indicators"]["kdj_period"],
            kdj_smooth=cfg["indicators"]["kdj_smooth"],
            boll_period=cfg["indicators"]["boll_period"],
            boll_std=cfg["indicators"]["boll_std"],
            atr_period=cfg["indicators"]["atr_period"],
        )

    print("Filtering by volatility ...")
    filtered = {c: df for c, df in stock_data.items()
                if passes_volatility(df, **vol_cfg)}
    print(f"  Volatility gate: {len(filtered)} / {len(stock_data)} passed")

    print("Scoring ...")
    picks = score_all(filtered,
                      threshold=score_cfg["threshold"],
                      require_all_dimensions=score_cfg.get("require_all_dimensions", True),
                      require_rsi_divergence=score_cfg.get("require_rsi_divergence", True),
                      rsi_long_oversold=score_cfg.get("rsi_long_oversold", 30))
    picks = picks[:n]
    print(f"  Selected: {len(picks)} stocks")

    # Fetch stock meta (name + industry)
    meta = _get_stock_meta(db_path)

    date_str = pick_date.replace("-", "")
    out_dir = cfg["pick"]["output_dir"]
    csv_path = os.path.join(out_dir, f"picks_{date_str}.csv")
    html_path = os.path.join(out_dir, f"report_{date_str}.html")

    picks_df = save_csv(picks, meta, csv_path)
    save_picks_html(picks, stock_data, meta, picks_df, html_path,
                    pick_date)

    print(f"\nDone. CSV: {csv_path}")
    print(f"HTML: {os.path.abspath(html_path)}")


def _get_stock_meta(db_path: str) -> dict:
    """Load stock name/industry from DB cache (written during ingest). No network call."""
    from src.data_fetcher import load_stock_basic
    return load_stock_basic(db_path)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="A股主板反转选股策略")
    parser.add_argument("command", nargs="?",
                        choices=["validate", "ingest", "backtest", "pick"],
                        default=None)
    parser.add_argument("--code", default="sh.600519")
    parser.add_argument("--since", default=None)
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--date", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.command == "validate":
        cmd_validate(cfg, args.code)
    elif args.command == "ingest":
        cmd_ingest(cfg, args.since)
    elif args.command == "backtest":
        cmd_backtest(cfg, args.auto_tune)
    elif args.command == "pick":
        cmd_pick(cfg, args.date, args.top)
    else:
        # Full pipeline
        print("Running full pipeline: validate → ingest → backtest → pick")
        cmd_validate(cfg)
        cmd_ingest(cfg)
        passed, cfg = cmd_backtest(cfg, auto_tune=True)
        cmd_pick(cfg)


if __name__ == "__main__":
    main()
