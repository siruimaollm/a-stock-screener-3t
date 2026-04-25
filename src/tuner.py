"""
Auto-tuner: grid search over strategy parameters when backtest fails.
两阶段网格搜索：先扫 amp30×threshold，再扫 atr_pct_ratio×bb_width_quantile。
"""
from itertools import product
from typing import Any, Optional, Dict

import pandas as pd

from src.backtest import run_backtest, check_pass_criteria


def grid_search(all_kline: pd.DataFrame,
                benchmark_kline: pd.DataFrame,
                criteria: dict,
                backtest_cfg: dict,
                tuner_cfg: dict,
                oos_kline: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    amp_candidates = tuner_cfg.get("amp30_candidates", [0.12, 0.15, 0.20])
    thr_candidates = tuner_cfg.get("score_threshold_candidates", [5, 6, 7])
    atr_ratio_candidates = tuner_cfg.get("atr_pct_ratio_candidates", [1.2, 1.5, 2.0])
    bw_q_candidates = tuner_cfg.get("bb_width_quantile_candidates", [0.6, 0.7, 0.8])

    hold_periods = backtest_cfg.get("hold_periods", [5, 10, 20])
    top_n = backtest_cfg.get("top_n", 20)
    start = backtest_cfg.get("start_date", "2025-07-01")
    end = backtest_cfg.get("end_date", "2026-04-25")
    oos_start = backtest_cfg.get("out_of_sample_start", "2025-04-01")
    oos_end = backtest_cfg.get("out_of_sample_end", "2025-06-30")

    all_trials = []
    best_config = None
    best_results = None
    best_score = -1

    # Phase 1: amp30 × threshold
    print("=== Tuner Phase 1: amp30 × threshold ===")
    for amp, thr in product(amp_candidates, thr_candidates):
        vol_cfg = {"amp30_min": amp, "atr_pct_ratio": 1.5,
                   "bb_width_quantile": 0.80, "lookback": 90}
        score_cfg = {"threshold": thr, "require_all_dimensions": True}
        print(f"  amp={amp} thr={thr} ...", end=" ")
        results = run_backtest(all_kline, benchmark_kline, start, end,
                               hold_periods, top_n, vol_cfg, score_cfg)
        passed, _ = check_pass_criteria(results, criteria)
        m10 = results.get("metrics_10", {})
        trial = {"amp30": amp, "threshold": thr,
                 "atr_pct_ratio": 1.5, "bb_width_quantile": 0.80,
                 "passed": passed, "metrics_10": m10,
                 "metrics_20": results.get("metrics_20", {})}
        all_trials.append(trial)
        exc_wr = m10.get("exc_win_rate", 0) or 0
        print(f"pass={passed} exc_wr={exc_wr:.1%}")
        if passed and exc_wr > best_score:
            best_score = exc_wr
            best_config = trial.copy()
            best_results = results

    if best_config:
        best_amp = best_config["amp30"]
        best_thr = best_config["threshold"]
    else:
        best_partial = max(all_trials, key=lambda x: x["metrics_10"].get("exc_win_rate", 0) or 0)
        best_amp = best_partial["amp30"]
        best_thr = best_partial["threshold"]

    # Phase 2: atr_pct_ratio × bb_width_quantile
    print(f"=== Tuner Phase 2: atr_ratio × bw_quantile (amp={best_amp} thr={best_thr}) ===")
    for ar, bq in product(atr_ratio_candidates, bw_q_candidates):
        if ar == 1.5 and bq == 0.80:
            continue
        vol_cfg = {"amp30_min": best_amp, "atr_pct_ratio": ar,
                   "bb_width_quantile": bq, "lookback": 90}
        score_cfg = {"threshold": best_thr, "require_all_dimensions": True}
        print(f"  atr_r={ar} bw_q={bq} ...", end=" ")
        results = run_backtest(all_kline, benchmark_kline, start, end,
                               hold_periods, top_n, vol_cfg, score_cfg)
        passed, _ = check_pass_criteria(results, criteria)
        m10 = results.get("metrics_10", {})
        exc_wr = m10.get("exc_win_rate", 0) or 0
        trial = {"amp30": best_amp, "threshold": best_thr,
                 "atr_pct_ratio": ar, "bb_width_quantile": bq,
                 "passed": passed, "metrics_10": m10,
                 "metrics_20": results.get("metrics_20", {})}
        all_trials.append(trial)
        print(f"pass={passed} exc_wr={exc_wr:.1%}")
        if passed and exc_wr > best_score:
            best_score = exc_wr
            best_config = trial.copy()
            best_results = results

    # OOS validation
    oos_passed = False
    oos_results = None
    if best_config and oos_kline is not None and not oos_kline.empty:
        print("=== Out-of-sample validation ===")
        vc = {"amp30_min": best_config["amp30"],
              "atr_pct_ratio": best_config["atr_pct_ratio"],
              "bb_width_quantile": best_config["bb_width_quantile"],
              "lookback": 90}
        sc = {"threshold": best_config["threshold"], "require_all_dimensions": True}
        oos_results = run_backtest(oos_kline, benchmark_kline, oos_start, oos_end,
                                   hold_periods, top_n, vc, sc)
        oos_passed, _ = check_pass_criteria(oos_results, criteria)
        print(f"OOS passed: {oos_passed}")

    return {
        "best_config": best_config,
        "best_results": best_results,
        "all_trials": all_trials,
        "oos_passed": oos_passed,
        "oos_results": oos_results,
    }
