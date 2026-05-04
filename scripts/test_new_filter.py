import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from src.data_fetcher import load_all_kline_sqlite
from src.indicators import add_all_indicators
from run import _build_vol_cfg

with open('config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

vol_cfg = _build_vol_cfg(cfg)
sqlite_path = cfg['data']['stock_data_db']

pick_date = '2026-04-30'
since = (datetime.strptime(pick_date, '%Y-%m-%d') - timedelta(days=220)).strftime('%Y-%m-%d')
all_kline = load_all_kline_sqlite(sqlite_path, table='daily_data_hfq',
                                  start_date=since, end_date=pick_date)

ind_cfg = {k: cfg['indicators'][k] for k in
           ['rsi_periods','kdj_period','kdj_smooth','boll_period','boll_std','atr_period']}

stock_data = {}
for code, grp in all_kline.groupby('code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    if len(grp) < 90:
        continue
    try:
        stock_data[code] = add_all_indicators(grp, **ind_cfg)
    except Exception:
        pass
print(f'指标完成: {len(stock_data)} 只\n')

# 详细漏斗
c = {f'cond{i}': 0 for i in range(1, 9)}
c['all'] = 0
rally_pcts = []

for code, df in stock_data.items():
    last = df.iloc[-1]
    close = float(last.get('close', np.nan))
    if pd.isna(close) or close < vol_cfg['min_close']:
        continue
    amt_ma5 = float(last.get('amount_ma5', np.nan))
    if pd.isna(amt_ma5) or amt_ma5 < vol_cfg['min_avg_amount']:
        continue
    c['cond1'] += 1   # 过了基础门槛

    rally_end = -(vol_cfg['consol_window'] - 5)
    rally_df = df.iloc[-vol_cfg['rally_lookback']: rally_end]
    if len(rally_df) < 10:
        continue
    rally_high = float(rally_df['high'].max())
    rally_low  = float(rally_df['low'].min())
    if rally_low <= 0:
        continue
    rally_pct = (rally_high - rally_low) / rally_low
    rally_pcts.append(rally_pct)
    if rally_pct < vol_cfg['rally_min_pct']:
        continue
    c['cond2'] += 1   # 过了涨幅≥25%

    vol_baseline = float(df['volume'].iloc[-90:].mean())
    vol_rally    = float(rally_df['volume'].mean())
    if vol_baseline <= 0 or vol_rally < vol_baseline * vol_cfg['rally_vol_ratio']:
        continue
    c['cond3'] += 1   # 过了放量1.3x

    lows  = rally_df['low'].values
    highs = rally_df['high'].values
    low_pos  = int(np.argmin(lows))
    highs_after = highs[low_pos:]
    high_pos = int(np.argmax(highs_after)) + low_pos
    if high_pos - low_pos < vol_cfg['rally_min_days']:
        continue
    c['cond4'] += 1   # 过了持续≥5天

    consol_df = df.iloc[-vol_cfg['consol_window']:]
    c_max  = float(consol_df['close'].max())
    c_min  = float(consol_df['close'].min())
    c_mean = float(consol_df['close'].mean())
    if c_mean <= 0 or (c_max - c_min) / c_mean > vol_cfg['range_max_pct']:
        continue
    c['cond5'] += 1   # 过了横盘≤12%

    high_60d = float(df['high'].iloc[-vol_cfg['rally_lookback']:].max())
    dist_ratio = close / high_60d if high_60d > 0 else 0
    if not (vol_cfg['drawdown_min'] <= dist_ratio <= vol_cfg['drawdown_max']):
        continue
    c['cond6'] += 1   # 过了回撤40%~80%

    vol_10d = float(df['volume'].iloc[-10:].mean())
    vol_30d = float(df['volume'].iloc[-30:].mean())
    if vol_30d <= 0 or vol_10d / vol_30d > vol_cfg['vol_shrink_ratio']:
        continue
    c['cond7'] += 1   # 过了缩量0.6x

    if 'atr' in df.columns:
        atr_10d = float(df['atr'].iloc[-10:].mean())
        atr_30d = float(df['atr'].iloc[-30:].mean())
        if atr_30d > 0 and atr_10d / atr_30d > vol_cfg['atr_shrink_ratio']:
            continue

    if len(df) >= 6:
        c5ago = float(df['close'].iloc[-6])
        if c5ago > 0 and (close - c5ago) / c5ago < vol_cfg['cum_5d_min']:
            continue
    c['cond8'] += 1   # 过了未急跌
    c['all']   += 1

total = len(stock_data)
print(f"{'条件':<30} {'通过':>6} {'占比':>7}")
print('-' * 45)
print(f"{'基础（价格+流动性）':<30} {c['cond1']:>6} {c['cond1']/total*100:>6.1f}%")
print(f"{'涨幅≥25%（60~25日前）':<30} {c['cond2']:>6} {c['cond2']/total*100:>6.1f}%")
print(f"{'放量≥均量×1.3':<30} {c['cond3']:>6} {c['cond3']/total*100:>6.1f}%")
print(f"{'上涨持续≥5天':<30} {c['cond4']:>6} {c['cond4']/total*100:>6.1f}%")
print(f"{'横盘区间≤12%（近30日）':<30} {c['cond5']:>6} {c['cond5']/total*100:>6.1f}%")
print(f"{'回撤40%~80%（vs 60日高）':<30} {c['cond6']:>6} {c['cond6']/total*100:>6.1f}%")
print(f"{'缩量≤0.6x（10日vs30日）':<30} {c['cond7']:>6} {c['cond7']/total*100:>6.1f}%")
print(f"{'全部通过':<30} {c['all']:>6} {c['all']/total*100:>6.1f}%")

if rally_pcts:
    arr = np.array(rally_pcts)
    print(f"\n涨幅分布（60~25日前窗口）:")
    print(f"  中位数={np.median(arr)*100:.1f}%  均值={np.mean(arr)*100:.1f}%")
    print(f"  ≥25%占比: {(arr>=0.25).mean()*100:.1f}%")
    print(f"  ≥15%占比: {(arr>=0.15).mean()*100:.1f}%")
    print(f"  ≥10%占比: {(arr>=0.10).mean()*100:.1f}%")
