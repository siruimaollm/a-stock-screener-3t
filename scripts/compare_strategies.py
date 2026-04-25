"""
对比两套策略：
  方案A: threshold=7, RSI12<35 (放宽中周期超卖阈值)
  方案B: threshold=6, RSI12<30 (PDF原始定义,降低总分门槛)
两者都保留 has_divergence 必要条件。
"""
import yaml, duckdb, pandas as pd
from src.backtest import run_backtest, check_pass_criteria

cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
db_path = cfg['data']['db_path']
bt_cfg = cfg['backtest']

conn = duckdb.connect(db_path, read_only=True)
all_kline = conn.execute(
    "SELECT * FROM daily_kline WHERE date >= ? AND date <= ? ORDER BY code, date",
    ['2025-04-01', bt_cfg['end_date']]).df()
try:
    bench_kline = conn.execute(
        "SELECT * FROM daily_kline WHERE code = ? AND date >= ? AND date <= ? ORDER BY date",
        [bt_cfg['benchmark'], bt_cfg['start_date'], bt_cfg['end_date']]).df()
except Exception:
    bench_kline = pd.DataFrame()
conn.close()

vol_cfg = {
    'amp30_min': cfg['volatility']['amp30_min'],
    'atr_pct_ratio': cfg['volatility']['atr_pct_ratio'],
    'bb_width_quantile': cfg['volatility']['bb_width_quantile'],
    'lookback': cfg['volatility'].get('lookback', 90),
}

scenarios = {
    "A: threshold=7, RSI12<35": {
        "threshold": 7,
        "require_all_dimensions": True,
        "require_rsi_divergence": True,
        "rsi_long_oversold": 35,
    },
    "B: threshold=6, RSI12<30": {
        "threshold": 6,
        "require_all_dimensions": True,
        "require_rsi_divergence": True,
        "rsi_long_oversold": 30,
    },
}

results_summary = {}

for name, score_cfg in scenarios.items():
    print(f"\n{'='*60}")
    print(f"Running scenario: {name}")
    print(f"  config: {score_cfg}")
    print('='*60)
    res = run_backtest(
        all_kline, bench_kline,
        bt_cfg['start_date'], bt_cfg['end_date'],
        bt_cfg['hold_periods'], bt_cfg['top_n'],
        vol_cfg, score_cfg
    )
    results_summary[name] = res

# Print comparison table
print("\n\n" + "="*80)
print("STRATEGY COMPARISON")
print("="*80)

for name, res in results_summary.items():
    print(f"\n{name}:")
    for h in [5, 10, 20]:
        m = res.get(f'metrics_{h}', {})
        n = m.get('count', 0)
        wr = m.get('win_rate', 0) or 0
        ret = m.get('mean_return', 0) or 0
        med = m.get('median_return', 0) or 0
        max_loss = m.get('max_loss', 0) or 0
        exc_wr = m.get('exc_win_rate', None)
        exc_str = f"{exc_wr*100:.1f}%" if exc_wr is not None and not pd.isna(exc_wr) else "n/a"
        print(f"  {h}日: n={n}  win={wr*100:.1f}%  mean={ret*100:+.2f}%  median={med*100:+.2f}%  maxloss={max_loss*100:.2f}%  exc_win={exc_str}")
