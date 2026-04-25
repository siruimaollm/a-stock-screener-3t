"""诊断: 在回测期内, 严格条件(threshold=7 + divergence required) 每天选出几只?"""
import yaml, duckdb, pandas as pd
from datetime import datetime, timedelta
from src.indicators import add_all_indicators
from src.volatility_filter import passes_volatility
from src.scoring import score_stock, is_selected

cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
conn = duckdb.connect(cfg['data']['db_path'], read_only=True)
all_kline = conn.execute(
    "SELECT * FROM daily_kline WHERE date >= ? ORDER BY code, date",
    ['2025-04-01']).df()
conn.close()

print(f"Loaded {len(all_kline)} rows for {all_kline['code'].nunique()} stocks")

# Pre-compute indicators
indicator_cache = {}
for code, grp in all_kline.groupby('code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    if len(grp) >= 60:
        indicator_cache[code] = add_all_indicators(grp)

print(f"Indicators computed for {len(indicator_cache)} stocks")

vol_cfg = {
    'amp30_min': cfg['volatility']['amp30_min'],
    'atr_pct_ratio': cfg['volatility']['atr_pct_ratio'],
    'bb_width_quantile': cfg['volatility']['bb_width_quantile'],
    'lookback': cfg['volatility'].get('lookback', 90),
}

# Sample test dates monthly through backtest period
test_dates = ['2025-07-15', '2025-08-15', '2025-09-15', '2025-10-15',
              '2025-11-17', '2025-12-15', '2026-01-15', '2026-02-13',
              '2026-03-13', '2026-04-15', '2026-04-25']

for t_date in test_dates:
    # gather candidates passing each layer
    n_vol = 0
    n_score7 = 0  # total >= 7 + dimensions OK (no div requirement)
    n_div = 0  # divergence detected (any score)
    n_final = 0  # all conditions: score>=7 + dimensions + divergence
    bolls, rsis, kdjs, divs = [], [], [], []

    for code, df_full in indicator_cache.items():
        df_t = df_full[df_full['date'] <= t_date]
        if len(df_t) < 30:
            continue
        if not passes_volatility(df_t, **vol_cfg):
            continue
        n_vol += 1
        s = score_stock(df_t)
        if s['has_divergence']:
            n_div += 1
            divs.append(code)
        # threshold=7, all dims, no div
        if (s['total_score'] >= 7 and s['boll_score'] >= 1
                and s['rsi_score'] >= 1 and s['kdj_score'] >= 1):
            n_score7 += 1
            bolls.append(s['boll_score'])
            rsis.append(s['rsi_score'])
            kdjs.append(s['kdj_score'])
        if is_selected(s, threshold=7, require_rsi_divergence=True):
            n_final += 1

    print(f"\n{t_date}: vol={n_vol}  score>=7={n_score7}  has_div={n_div}  FINAL={n_final}")
    if n_score7 > 0:
        print(f"  score>=7 stocks BOLL avg={sum(bolls)/len(bolls):.1f} RSI avg={sum(rsis)/len(rsis):.1f} KDJ avg={sum(kdjs)/len(kdjs):.1f}")
