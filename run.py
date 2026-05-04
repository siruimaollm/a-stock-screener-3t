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
import json
import os
import re
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
    from src.data_fetcher import update_stock_data

    print("\n=== STEP 2: 增量更新 stock_data.db ===")
    stock_db = cfg["data"].get("stock_data_db", "data/stock_data.db")
    today = datetime.today().strftime("%Y-%m-%d")

    print("Logging in to BaoStock ...")
    lg = bs.login()
    if lg.error_code != "0":
        print(f"BaoStock login failed: {lg.error_msg}")
        sys.exit(1)

    stats = update_stock_data(
        db_path=stock_db,
        end_date=since or today,
        workers=cfg["data"].get("baostock_workers", 4),
        retry=cfg["data"].get("retry", 3),
    )

    bs.logout()
    print(f"\n更新结果: {stats}")


# ─────────────────────────────────────────────
#  BACKTEST
# ─────────────────────────────────────────────
def cmd_backtest(cfg: dict, auto_tune: bool = False):
    from src.backtest import run_backtest, check_pass_criteria
    from src.backtest_report import save_backtest_html
    from src.data_fetcher import load_all_kline_sqlite, load_benchmark_kline

    print("\n=== STEP 3: Backtest ===")
    stock_db = cfg["data"].get("stock_data_db", "data/stock_data.db")
    bt_cfg = cfg["backtest"]
    pass_criteria = bt_cfg["pass_criteria"]
    hold_periods = bt_cfg["hold_periods"]

    # 回测需要指标预热，多加载 220 天历史
    from datetime import datetime as _dt, timedelta as _td
    hist_start = (_dt.strptime(bt_cfg["start_date"], "%Y-%m-%d") - _td(days=220)).strftime("%Y-%m-%d")

    print(f"Loading main board kline ({hist_start} ~ {bt_cfg['end_date']}) ...")
    all_kline = load_all_kline_sqlite(stock_db, table="daily_data_hfq",
                                      start_date=hist_start,
                                      end_date=bt_cfg["end_date"])
    print(f"  {len(all_kline)} rows, {all_kline['code'].nunique()} stocks")

    # 基准指数：从 index_data 表读取（代码去掉 sh./sz. 前缀）
    bench_code_raw = bt_cfg["benchmark"]
    bench_code = bench_code_raw.split(".")[-1]   # 'sh.000300' → '000300'
    print(f"Loading benchmark {bench_code} from index_data ...")
    bench_kline = load_benchmark_kline(stock_db, index_code=bench_code,
                                       start_date=bt_cfg["start_date"],
                                       end_date=bt_cfg["end_date"])
    if bench_kline.empty:
        print("  Benchmark not found in index_data, fetching via BaoStock ...")
        bench_kline = _fetch_benchmark(bench_code_raw, bt_cfg["start_date"], bt_cfg["end_date"])

    vol_cfg = _build_vol_cfg(cfg)
    score_cfg = {"threshold": cfg["scoring"]["threshold"],
                 "require_all_dimensions": cfg["scoring"].get("require_all_dimensions", True),
                 "require_rsi_divergence": cfg["scoring"].get("require_rsi_divergence", False),
                 "rsi_long_oversold": cfg["scoring"].get("rsi_long_oversold", 40)}
    ind_cfg = {
        "rsi_periods": cfg["indicators"]["rsi_periods"],
        "kdj_period":  cfg["indicators"]["kdj_period"],
        "kdj_smooth":  cfg["indicators"]["kdj_smooth"],
        "boll_period": cfg["indicators"]["boll_period"],
        "boll_std":    cfg["indicators"]["boll_std"],
        "atr_period":  cfg["indicators"]["atr_period"],
    }

    results = run_backtest(all_kline, bench_kline,
                           bt_cfg["start_date"], bt_cfg["end_date"],
                           hold_periods, bt_cfg["top_n"],
                           vol_cfg, score_cfg, ind_cfg)

    passed, failed = check_pass_criteria(results, pass_criteria)
    auto_tuned = False

    if not passed and auto_tune:
        print(f"\n*** Backtest failed: {failed} ***")
        print("Starting auto-tune grid search ...")
        from src.tuner import grid_search

        oos_kline = load_all_kline_sqlite(stock_db, table="daily_data_hfq",
                                          start_date=bt_cfg["out_of_sample_start"],
                                          end_date=bt_cfg["out_of_sample_end"])

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
    from src.data_fetcher import load_all_kline_sqlite
    from src.indicators import add_all_indicators
    from src.volatility_filter import passes_volatility
    from src.scoring import score_all
    from src.report_csv import save_csv
    from src.report_html import save_picks_html

    print("\n=== STEP 4: Pick stocks ===")

    if pick_date is None:
        pick_date = _last_trading_date()
    n = top_n or cfg["pick"]["top_n"]
    sqlite_path = cfg["data"].get("stock_data_db", "data/stock_data.db")

    # need ~120 bars before pick_date for indicators
    from datetime import datetime as dt, timedelta
    since_dt = dt.strptime(pick_date, "%Y-%m-%d") - timedelta(days=200)
    since = since_dt.strftime("%Y-%m-%d")

    print(f"加载行情数据（{since} ~ {pick_date}）...")
    all_kline = load_all_kline_sqlite(sqlite_path, table="daily_data_hfq",
                                      start_date=since, end_date=pick_date)

    if all_kline.empty:
        print("No data in stock_data.db. Please check the database.")
        return

    vol_cfg = _build_vol_cfg(cfg)
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
    require_div = score_cfg.get("require_rsi_divergence", True)
    divergence_relaxed = False

    picks = score_all(filtered,
                      threshold=score_cfg["threshold"],
                      require_all_dimensions=score_cfg.get("require_all_dimensions", True),
                      require_rsi_divergence=require_div,
                      rsi_long_oversold=score_cfg.get("rsi_long_oversold", 30))

    # 严格模式结果过少时（<5），自动放开 RSI 底背离要求
    if len(picks) < 5 and require_div:
        print(f"  [严格模式] 仅 {len(picks)} 只入选，放开 RSI 底背离条件后重试 ...")
        picks = score_all(filtered,
                          threshold=score_cfg["threshold"],
                          require_all_dimensions=score_cfg.get("require_all_dimensions", True),
                          require_rsi_divergence=False,
                          rsi_long_oversold=score_cfg.get("rsi_long_oversold", 30))
        divergence_relaxed = True

    picks = picks[:n]
    mode_label = "宽松(无底背离要求)" if divergence_relaxed else "严格(含底背离)"
    print(f"  Selected: {len(picks)} stocks  [模式: {mode_label}]")

    # 给每条结果标注选股模式，方便报告展示
    for p in picks:
        p["mode"] = mode_label

    # Fetch stock meta (name + industry)
    meta = _get_stock_meta(cfg)

    date_str = pick_date.replace("-", "")
    out_dir = cfg["pick"]["output_dir"]
    csv_path = os.path.join(out_dir, f"picks_{date_str}.csv")
    html_path = os.path.join(out_dir, f"report_{date_str}.html")

    picks_df = save_csv(picks, meta, csv_path)
    save_picks_html(picks, stock_data, meta, picks_df, html_path,
                    pick_date)

    print(f"\nDone. CSV: {csv_path}")
    print(f"HTML: {os.path.abspath(html_path)}")


def cmd_backtest_v2(cfg: dict):
    """v2 回测：TP=+10%或布林上轨，SL=破位-8%或持有1年。"""
    from src.data_fetcher import load_all_kline_sqlite, load_benchmark_kline
    from src.backtest_v2 import run_backtest_v2
    from src.backtest_report import save_backtest_v2_html
    from datetime import datetime as _dt, timedelta as _td

    print("\n=== STEP 3 (v2): 优化版历史回测 ===")
    stock_db = cfg["data"].get("stock_data_db", "data/stock_data.db")
    bt_cfg = cfg["backtest"]

    hist_start = (_dt.strptime(bt_cfg["start_date"], "%Y-%m-%d") - _td(days=220)).strftime("%Y-%m-%d")

    print(f"加载行情 ({hist_start} ~ {bt_cfg['end_date']}) ...")
    all_kline = load_all_kline_sqlite(stock_db, table="daily_data_hfq",
                                      start_date=hist_start,
                                      end_date=bt_cfg["end_date"])
    print(f"  {len(all_kline):,} 行, {all_kline['code'].nunique()} 只股票")

    bench_code = bt_cfg["benchmark"].split(".")[-1]
    print(f"加载基准 {bench_code} ...")
    bench_kline = load_benchmark_kline(stock_db, index_code=bench_code,
                                       start_date=bt_cfg["start_date"],
                                       end_date=bt_cfg["end_date"])

    vol_cfg = _build_vol_cfg(cfg)
    score_cfg = {
        "threshold":              cfg["scoring"]["threshold"],
        "require_all_dimensions": cfg["scoring"].get("require_all_dimensions", True),
        "require_rsi_divergence": cfg["scoring"].get("require_rsi_divergence", False),
        "rsi_long_oversold":      cfg["scoring"].get("rsi_long_oversold", 40),
    }
    ind_cfg = {
        "rsi_periods": cfg["indicators"]["rsi_periods"],
        "kdj_period":  cfg["indicators"]["kdj_period"],
        "kdj_smooth":  cfg["indicators"]["kdj_smooth"],
        "boll_period": cfg["indicators"]["boll_period"],
        "boll_std":    cfg["indicators"]["boll_std"],
        "atr_period":  cfg["indicators"]["atr_period"],
    }
    bt_v2_cfg = cfg.get("backtest_v2", {})
    take_profit    = bt_v2_cfg.get("take_profit",    0.10)
    break_down_pct = bt_v2_cfg.get("break_down_pct", 0.08)
    max_hold       = bt_v2_cfg.get("max_hold",       252)
    signal_freq    = bt_v2_cfg.get("signal_freq",    "biweekly")

    result = run_backtest_v2(
        all_kline, bench_kline,
        bt_cfg["start_date"], bt_cfg["end_date"],
        top_n=bt_cfg["top_n"],
        vol_cfg=vol_cfg, score_cfg=score_cfg, ind_cfg=ind_cfg,
        take_profit=take_profit, break_down_pct=break_down_pct,
        max_hold=max_hold, signal_freq=signal_freq,
    )

    metrics = result["metrics"]
    trades_df = result["trades"]

    out_dir = cfg["pick"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.today().strftime("%Y%m%d")
    out_html = os.path.join(out_dir, f"backtest_v2_{date_str}.html")
    out_csv  = os.path.join(out_dir, f"backtest_v2_{date_str}_trades.csv")

    if not trades_df.empty:
        # 注入股票名称 / 行业（从 stock_info 表）
        meta = _get_stock_meta(cfg)
        trades_df = trades_df.copy()
        # 确保 code 为 6 位字符串（防止 CSV 读取时丢失前导 0）
        trades_df["code"] = trades_df["code"].astype(str).str.zfill(6)
        trades_df["name"]     = trades_df["code"].map(lambda c: (meta.get(c, {}) or {}).get("name", ""))
        trades_df["industry"] = trades_df["code"].map(lambda c: (meta.get(c, {}) or {}).get("industry", ""))
        # 重新排序列：把 name/industry 放到 code 后面
        cols = list(trades_df.columns)
        if "code" in cols and "name" in cols:
            cols.remove("name");  cols.insert(cols.index("code") + 1, "name")
        if "industry" in cols:
            cols.remove("industry"); cols.insert(cols.index("name") + 1, "industry")
        trades_df = trades_df[cols]

        trades_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        save_backtest_v2_html(metrics, trades_df, out_html,
                              bt_cfg["start_date"], bt_cfg["end_date"],
                              meta=meta)

    print()
    print("=" * 56)
    print(f"[v2] 总笔数={metrics.get('total_trades',0)}")
    if metrics.get("total_trades", 0) > 0:
        print(f"[v2] 胜率={metrics['win_rate']:.1%}  止盈={metrics['tp_rate']:.1%} (+10%:{metrics['tp_pct_rate']:.1%} 布林:{metrics['tp_boll_rate']:.1%})")
        print(f"[v2] 破位止损={metrics['sl_rate']:.1%}  超时(1y)={metrics['timeout_rate']:.1%}")
        print(f"[v2] 平均收益={metrics['mean_return']:.2%}  中位={metrics['median_return']:.2%}  最优={metrics['max_return']:.2%}  最差={metrics['min_return']:.2%}")
        print(f"[v2] 超额胜率={metrics.get('exc_win_rate', 0):.1%}  平均超额={metrics.get('mean_excess', 0):.2%}")
        print(f"[v2] 平均持有={metrics['mean_hold_days']:.1f} 日")
        print(f"[v2] HTML报告: {out_html}")
        print(f"[v2] 明细CSV : {out_csv}")
    print("=" * 56)


def cmd_backtest_tp(cfg: dict):
    """10%止盈/5%止损回测。"""
    from src.data_fetcher import load_all_kline_sqlite
    from src.backtest_tp import run_tp_backtest

    print("\n=== 止盈止损回测（+10% / -5%）===")
    sqlite_path = cfg["data"].get("stock_data_db", "data/stock_data.db")
    bt_cfg  = cfg.get("backtest", {})
    start   = bt_cfg.get("start_date", "2025-07-01")
    end     = bt_cfg.get("end_date",   "2026-04-25")

    print(f"加载 {start} ~ {end} 全市场日线（5年后复权）...")
    from datetime import datetime, timedelta
    hist_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=220)).strftime("%Y-%m-%d")
    all_kline = load_all_kline_sqlite(sqlite_path, table="daily_data_hfq",
                                      start_date=hist_start, end_date=end)
    print(f"  共 {len(all_kline)} 行，{all_kline['code'].nunique()} 只股票")

    vol_cfg = _build_vol_cfg(cfg)
    score_cfg = {
        "threshold":             cfg["scoring"]["threshold"],
        "require_all_dimensions": cfg["scoring"].get("require_all_dimensions", True),
        "require_rsi_divergence": cfg["scoring"].get("require_rsi_divergence", False),
        "rsi_long_oversold":      cfg["scoring"].get("rsi_long_oversold", 40),
    }
    ind_cfg = {
        "rsi_periods": cfg["indicators"]["rsi_periods"],
        "kdj_period":  cfg["indicators"]["kdj_period"],
        "kdj_smooth":  cfg["indicators"]["kdj_smooth"],
        "boll_period": cfg["indicators"]["boll_period"],
        "boll_std":    cfg["indicators"]["boll_std"],
        "atr_period":  cfg["indicators"]["atr_period"],
    }

    result = run_tp_backtest(
        all_kline, vol_cfg, score_cfg, ind_cfg,
        start_date=start, end_date=end,
        take_profit=0.10, stop_loss=0.05,
        max_hold=60, signal_freq="biweekly",
    )

    if result.get("total_trades", 0) == 0:
        print("无有效交易，请检查策略参数或扩大回测区间。")
        return

    print(f"\n{'='*50}")
    print(f"回测区间: {start} ~ {end}")
    print(f"总交易笔数:   {result['total_trades']}")
    print(f"胜率(盈利):   {result['win_rate']:.1%}")
    print(f"  止盈触发:   {result['tp_rate']:.1%}  ({result['take_profit']:.0%})")
    print(f"  止损触发:   {result['sl_rate']:.1%}  (-{result['stop_loss']:.0%})")
    print(f"  超时平仓:   {result['timeout_rate']:.1%}")
    print(f"平均收益:     {result['mean_return']:.2%}")
    print(f"中位数收益:   {result['median_return']:.2%}")
    print(f"最佳单笔:     {result['max_return']:.2%}")
    print(f"最差单笔:     {result['min_return']:.2%}")
    print(f"平均持有天数: {result['mean_hold_days']:.1f} 日")
    print(f"{'='*50}")

    # 保存明细
    out_dir = cfg["pick"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "backtest_tp_detail.csv")
    result["trades_df"].to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"明细已保存: {detail_path}")


def cmd_backtest_vbt(cfg: dict):
    """VectorBT 向量化止盈止损回测（比 backtest-tp 快 10-20x）。"""
    from src.data_fetcher import load_all_kline_sqlite
    from src.backtest_vbt import run_vbt_backtest

    print("\n=== VectorBT 向量化回测（+10% / -5%）===")
    sqlite_path = cfg["data"].get("stock_data_db", "data/stock_data.db")
    bt_cfg  = cfg.get("backtest", {})
    start   = bt_cfg.get("start_date", "2025-07-01")
    end     = bt_cfg.get("end_date",   "2026-04-25")

    print(f"加载 {start} ~ {end} 全市场日线...")
    from datetime import datetime, timedelta
    hist_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=220)).strftime("%Y-%m-%d")
    all_kline = load_all_kline_sqlite(sqlite_path, table="daily_data_hfq",
                                      start_date=hist_start, end_date=end)
    print(f"  共 {len(all_kline)} 行，{all_kline['code'].nunique()} 只股票")

    vol_cfg = _build_vol_cfg(cfg)
    score_cfg = {
        "threshold":              cfg["scoring"]["threshold"],
        "require_all_dimensions": cfg["scoring"].get("require_all_dimensions", True),
        "require_rsi_divergence": cfg["scoring"].get("require_rsi_divergence", False),
        "rsi_long_oversold":      cfg["scoring"].get("rsi_long_oversold", 40),
    }
    ind_cfg = {
        "rsi_periods": cfg["indicators"]["rsi_periods"],
        "kdj_period":  cfg["indicators"]["kdj_period"],
        "kdj_smooth":  cfg["indicators"]["kdj_smooth"],
        "boll_period": cfg["indicators"]["boll_period"],
        "boll_std":    cfg["indicators"]["boll_std"],
        "atr_period":  cfg["indicators"]["atr_period"],
    }

    result = run_vbt_backtest(
        all_kline, vol_cfg, score_cfg, ind_cfg,
        start_date=start, end_date=end,
        take_profit=0.10, stop_loss=0.10,
        max_hold=60, max_positions=5,
        signal_freq="biweekly",
    )

    if result.get("total_trades", 0) == 0:
        print("无有效交易，请检查策略参数或扩大回测区间。")
        return

    print(f"\n{'='*55}")
    print(f"回测区间: {start} ~ {end}")
    print(f"总交易笔数:      {result['total_trades']}")
    print(f"胜率(盈利):      {result['win_rate']:.1%}")
    print(f"  止盈合计:      {result['tp_rate']:.1%}")
    print(f"    └ +10%止盈:  {result.get('tp_pct_rate', 0):.1%}")
    print(f"    └ 布林上轨:  {result.get('tp_boll_rate', 0):.1%}")
    print(f"  止损(-10%):    {result['sl_rate']:.1%}")
    print(f"  超时平仓:      {result['timeout_rate']:.1%}")
    print(f"平均收益:        {result['mean_return']:.2%}")
    print(f"中位数收益:      {result['median_return']:.2%}")
    print(f"最佳单笔:        {result['max_return']:.2%}")
    print(f"最差单笔:        {result['min_return']:.2%}")
    print(f"平均持有天数:    {result['mean_hold_days']:.1f} 日")
    print(f"{'='*55}")

    out_dir = cfg["pick"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "backtest_vbt_detail.csv")
    result["trades_df"].to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"明细已保存: {detail_path}")


# ─────────────────────────────────────────────
#  妙想 API 命令
# ─────────────────────────────────────────────

def _miao_apikey(cfg: dict) -> Optional[str]:
    """从 config.yaml 或环境变量获取 MX_APIKEY。"""
    return cfg.get("miao", {}).get("api_key") or os.getenv("MX_APIKEY") or None


def cmd_miao_data(cfg: dict, question: str):
    """妙想金融数据查询。"""
    from src.miao_api import MiaoData

    api_key = _miao_apikey(cfg)
    client = MiaoData(api_key=api_key)

    print(f"\n[妙想金融数据] 查询: {question}")
    result = client.query(question)
    tables, err = client.to_dataframes(result)

    if err:
        print(f"错误: {err}")
        raw = json.dumps(result, ensure_ascii=False)[:800]
        print(f"原始响应片段: {raw}")
        return

    out_dir = cfg["pick"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for t in tables:
        df = t["df"]
        print(f"\n── {t['name']} ({len(df)} 行) ──")
        print(df.to_string(index=False, max_rows=20))

        # 保存 CSV
        safe = re.sub(r'[<>:"/\\|?*\[\] ]', "_", question)[:40]
        csv_path = os.path.join(out_dir, f"miao_data_{safe}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  已保存: {csv_path}")

    # 保存原始 JSON
    safe = re.sub(r'[<>:"/\\|?*\[\] ]', "_", question)[:40]
    json_path = os.path.join(out_dir, f"miao_data_{safe}_raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"原始JSON: {json_path}")


def cmd_miao_search(cfg: dict, question: str):
    """妙想资讯搜索。"""
    from src.miao_api import MiaoSearch

    api_key = _miao_apikey(cfg)
    client = MiaoSearch(api_key=api_key)

    print(f"\n[妙想资讯搜索] 查询: {question}")
    result = client.search(question)
    text = client.to_text(result)
    print(text)

    out_dir = cfg["pick"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r'[<>:"/\\|?*\[\] ]', "_", question)[:40]
    txt_path = os.path.join(out_dir, f"miao_search_{safe}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n已保存: {txt_path}")


def cmd_miao_xuangu(cfg: dict, condition: str):
    """妙想智能选股。"""
    from src.miao_api import MiaoXuangu

    api_key = _miao_apikey(cfg)
    client = MiaoXuangu(api_key=api_key)

    print(f"\n[妙想智能选股] 条件: {condition}")
    result = client.screen(condition)
    df, total, err = client.to_dataframe(result)

    if err:
        print(f"错误: {err}")
        raw = json.dumps(result, ensure_ascii=False)[:800]
        print(f"原始响应片段: {raw}")
        return

    print(f"\n符合条件股票: {total} 只，返回 {len(df)} 行")
    if not df.empty:
        print(df.to_string(index=False, max_rows=30))

    out_dir = cfg["pick"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r'[<>:"/\\|?*\[\] ]', "_", condition)[:40]
    csv_path = os.path.join(out_dir, f"miao_xuangu_{safe}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {csv_path}")

    json_path = os.path.join(out_dir, f"miao_xuangu_{safe}_raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"原始JSON: {json_path}")


def _build_vol_cfg(cfg: dict) -> dict:
    """从 config.yaml 的 volatility 节构建 passes_volatility 参数字典。"""
    v = cfg["volatility"]
    return {
        # 基础门槛
        "min_close":        v.get("min_close", 3.0),
        "min_avg_amount":   v.get("min_avg_amount", 5e7),
        # 第一段：历史曾大涨
        "rally_lookback":   v.get("rally_lookback", 60),
        "rally_min_pct":    v.get("rally_min_pct", 0.25),
        "rally_vol_ratio":  v.get("rally_vol_ratio", 1.3),
        "rally_min_days":   v.get("rally_min_days", 5),
        # 第二段：缩量横盘
        "consol_window":    v.get("consol_window", 30),
        "range_max_pct":    v.get("range_max_pct", 0.12),
        "drawdown_min":     v.get("drawdown_min", 0.40),
        "drawdown_max":     v.get("drawdown_max", 0.80),
        "vol_shrink_ratio": v.get("vol_shrink_ratio", 0.6),
        "atr_shrink_ratio": v.get("atr_shrink_ratio", 0.75),
        # 第三段：未急跌
        "cum_5d_min":       v.get("cum_5d_min", -0.10),
    }


def _get_stock_meta(cfg: dict) -> dict:
    """Load stock name/industry. Prefers new stock_data.db, falls back to BaoStock cache."""
    sqlite_path = cfg["data"].get("stock_data_db", "data/stock_data.db")
    if os.path.exists(sqlite_path):
        from src.data_fetcher import load_stock_info_sqlite
        return load_stock_info_sqlite(sqlite_path)
    from src.data_fetcher import load_stock_basic
    return load_stock_basic(cfg["data"]["db_path"])


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="A股主板反转选股策略")
    parser.add_argument("command", nargs="?",
                        choices=["validate", "ingest", "backtest", "backtest-v2", "pick",
                                 "backtest-tp", "backtest-vbt",
                                 "miao-data", "miao-search", "miao-xuangu"],
                        default=None)
    parser.add_argument("--code", default="sh.600519")
    parser.add_argument("--since", default=None)
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--date", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", default=None,
                        help="妙想查询问句（用于 miao-data / miao-search / miao-xuangu）")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.command == "validate":
        cmd_validate(cfg, args.code)
    elif args.command == "ingest":
        cmd_ingest(cfg, args.since)
    elif args.command == "backtest":
        cmd_backtest(cfg, args.auto_tune)
    elif args.command == "backtest-v2":
        cmd_backtest_v2(cfg)
    elif args.command == "pick":
        cmd_pick(cfg, args.date, args.top)
    elif args.command == "backtest-tp":
        cmd_backtest_tp(cfg)
    elif args.command == "backtest-vbt":
        cmd_backtest_vbt(cfg)
    elif args.command == "miao-data":
        q = args.query or input("请输入查询问句: ")
        cmd_miao_data(cfg, q)
    elif args.command == "miao-search":
        q = args.query or input("请输入搜索问句: ")
        cmd_miao_search(cfg, q)
    elif args.command == "miao-xuangu":
        q = args.query or input("请输入选股条件: ")
        cmd_miao_xuangu(cfg, q)
    else:
        # Full pipeline
        print("Running full pipeline: validate → ingest → backtest → pick")
        cmd_validate(cfg)
        cmd_ingest(cfg)
        passed, cfg = cmd_backtest(cfg, auto_tune=True)
        cmd_pick(cfg)


if __name__ == "__main__":
    main()
