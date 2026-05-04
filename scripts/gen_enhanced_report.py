# -*- coding: utf-8 -*-
"""Generate enhanced pick report with near-miss analysis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml
import pandas as pd
from datetime import datetime, timedelta

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
from src.data_fetcher import load_all_kline_sqlite, load_stock_info_sqlite
from src.indicators import add_all_indicators
from src.volatility_filter import passes_volatility
from src.scoring import score_stock, is_selected
from src.report_html import _kline_chart

# Use last normal trading day (before holiday distortion)
pick_date = "2026-04-30"
since = (datetime.strptime(pick_date, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")

v = cfg["volatility"]
vol_cfg = {
    "min_close": v.get("min_close", 3.0), "min_avg_amount": v.get("min_avg_amount", 5e7),
    "rally_lookback": v.get("rally_lookback", 60), "rally_min_pct": v.get("rally_min_pct", 0.25),
    "rally_vol_ratio": v.get("rally_vol_ratio", 1.3), "rally_min_days": v.get("rally_min_days", 5),
    "consol_window": v.get("consol_window", 30), "range_max_pct": v.get("range_max_pct", 0.20),
    "drawdown_min": v.get("drawdown_min", 0.40), "drawdown_max": v.get("drawdown_max", 0.90),
    "vol_shrink_ratio": v.get("vol_shrink_ratio", 0.80), "atr_shrink_ratio": v.get("atr_shrink_ratio", 0.85),
    "cum_5d_min": v.get("cum_5d_min", -0.10),
}
threshold = cfg["scoring"]["threshold"]
rsi_oversold = cfg["scoring"].get("rsi_long_oversold", 40)
meta = load_stock_info_sqlite("data/stock_data.db")

print("Loading kline data...")
all_kline = load_all_kline_sqlite("data/stock_data.db", start_date=since, end_date=pick_date)
print(f"  {len(all_kline)} rows, {all_kline['code'].nunique()} stocks")

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

print("Running volatility filter...")
filtered = {c: df for c, df in stock_data.items() if passes_volatility(df, **vol_cfg)}
print(f"  {len(filtered)} / {len(stock_data)} passed")

print("Scoring...")
scored = []
for code, df in filtered.items():
    s = score_stock(df, rsi_long_oversold=rsi_oversold)
    s["code"] = code
    s["selected"] = is_selected(s, threshold)
    scored.append(s)
scored.sort(key=lambda x: x["total_score"], reverse=True)

strict_picks = [s for s in scored if s["selected"]]
print(f"  Strict picks (threshold={threshold}): {len(strict_picks)}")

# Generate charts for top 10
print("Generating charts...")
chart_htmls = []
for s in scored[:10]:
    code = s["code"]
    info = meta.get(code, {})
    name = info.get("name", code)
    df = stock_data.get(code)
    if df is None or len(df) < 14:
        continue
    try:
        chart = _kline_chart(df, title=f"{name}({code})", n_bars=60)
        chart_htmls.append((code, name, s, chart.render_embed()))
    except Exception as e:
        print(f"  chart error {code}: {e}")
print(f"  {len(chart_htmls)} charts generated")

# Build table rows
rows_html = ""
for i, s in enumerate(scored[:20]):
    code = s["code"]
    info = meta.get(code, {})
    name = info.get("name", code)
    industry = info.get("industry", "")
    v_ok = ((s["vol_score"] >= 2) or (s["candle_score"] >= 3)
            or (s["vol_score"] >= 1 and s["candle_score"] >= 2))

    if s["selected"]:
        sel_icon = "&#x2714; 入选"
        bg = "#d4edda"
    else:
        sel_icon = "&#x25B3; 观察"
        bg = "#fff3cd" if s["total_score"] >= 5 else "#fff8f8"

    fail_parts = []
    if s["boll_score"] < 1:
        fail_parts.append("无BOLL下轨触及")
    if not v_ok:
        fail_parts.append(f"量价确认不足(vol={s['vol_score']},K线={s['candle_score']})")
    if s["total_score"] < threshold:
        fail_parts.append(f"总分{s['total_score']}&lt;{threshold}")
    fail_str = " / ".join(fail_parts) if fail_parts else "通过全部条件"

    close_v = f"{float(s['close']):.2f}" if s["close"] == s["close"] else "-"
    rsi6_v = f"{float(s['rsi6']):.1f}" if s["rsi6"] == s["rsi6"] else "-"
    j_v = f"{float(s['J']):.1f}" if s["J"] == s["J"] else "-"
    pct_b_v = f"{float(s['pct_b']):.3f}" if s["pct_b"] == s["pct_b"] else "-"
    sigs = "; ".join(s.get("signals", []))

    rows_html += f"""<tr style="background:{bg}">
      <td>{i+1}</td>
      <td style="font-weight:bold;color:{'green' if s['selected'] else '#e67e22'}">{sel_icon}</td>
      <td><b>{code}</b></td><td>{name}</td><td>{industry}</td>
      <td style="text-align:center"><b>{s['total_score']}</b>/15</td>
      <td style="text-align:center">{s['boll_score']}</td>
      <td style="text-align:center">{s['rsi_score']}</td>
      <td style="text-align:center">{s['kdj_score']}</td>
      <td style="text-align:center">{s['vol_score']}</td>
      <td style="text-align:center">{s['candle_score']}</td>
      <td>{close_v}</td><td>{rsi6_v}</td><td>{j_v}</td><td>{pct_b_v}</td>
      <td style="font-size:11px;color:#555">{fail_str}</td>
      <td style="font-size:10px;color:#444">{sigs}</td>
    </tr>\n"""

# Build charts section
charts_section = ""
for code, name, s, embed in chart_htmls:
    total = s["total_score"]
    sigs = "; ".join(s.get("signals", []))
    color = "#27ae60" if s["selected"] else "#e67e22"
    label = "严格入选" if s["selected"] else "观察候选"
    charts_section += f"""
<details style="margin:8px 0;border:1px solid #ddd;border-radius:4px">
  <summary style="padding:10px;cursor:pointer;background:#f8f8f8;font-weight:bold">
    <span style="color:{color}">[{label} {total}/15分]</span> {name} ({code})
    <span style="color:#888;font-size:11px;margin-left:12px">{sigs}</span>
  </summary>
  <div style="padding:8px">{embed}</div>
</details>"""

# Note for 0 strict picks
note = ""
if len(strict_picks) == 0:
    note = f"""<div style="background:#fff3cd;border-left:4px solid #ffc107;padding:14px;border-radius:4px;margin:12px 0">
  <b>当前无严格入选（threshold={threshold}/15）</b><br><br>
  当前市场环境下符合策略入选条件的股票较少。下方表格显示通过三段波动率过滤的全部候选，
  以及未严格入选的"观察名单"。如需放宽，可在 config.yaml 中调整 <code>threshold</code>。
</div>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>A股主板反转选股 {pick_date}</title>
<style>
body {{ font-family: Microsoft YaHei, sans-serif; margin: 20px; background: #f9f9f9; }}
h1 {{ color: #2c3e50; }}
h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
.summary {{ background: #eaf4fb; padding: 12px; border-radius: 6px; margin: 10px 0; line-height: 1.8; }}
.legend {{ background: #f8f8f8; padding: 10px; font-size: 12px; border: 1px solid #ddd;
           border-radius: 4px; margin: 5px 0; line-height: 1.8; }}
table {{ border-collapse: collapse; font-size: 12px; width: 100%; margin: 10px 0; }}
th {{ background: #2c3e50; color: white; padding: 7px 8px; }}
td {{ padding: 5px 8px; border: 1px solid #e0e0e0; }}
tr:hover {{ filter: brightness(0.95); }}
code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
</style>
</head>
<body>
<h1>A股主板 RSI+KDJ+布林三指标共振反转策略 — 选股报告</h1>
<div class="summary">
  <b>数据截止：</b>{pick_date}（五一节前最后交易日）&nbsp;|&nbsp;
  <b>严格入选（total≥{threshold}）：</b><b style="color:{'green' if len(strict_picks)>0 else '#e67e22'}">{len(strict_picks)} 只</b>&nbsp;|&nbsp;
  <b>通过波动率过滤：</b>{len(filtered)} / {len(stock_data)} 只&nbsp;|&nbsp;
  <b>生成时间：</b>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
<div class="legend">
  <b>策略三层过滤：</b><br>
  ① <b>三段波动率过滤</b>：历史(60~25日前)曾大涨放量 → 近30日缩量横盘 → 近5日未急跌<br>
  ② <b>五维打分</b>（满分15）：BOLL触下轨(0~3) + RSI超卖(0~3) + KDJ金叉(0~3) + 量价止跌(0~3) + 蜡烛形态(0~3)<br>
  ③ <b>入选门槛</b>：BOLL≥1（必须）+ 量价确认（vol≥2 或 K线≥3 或 vol≥1+K线≥2）+ 总分≥{threshold}
</div>
{note}
<h2>候选股汇总（通过波动率过滤，按总分排序，显示前20）</h2>
<table>
<thead><tr>
  <th>序</th><th>状态</th><th>代码</th><th>名称</th><th>行业</th>
  <th>总分</th><th>BOLL</th><th>RSI</th><th>KDJ</th><th>量价</th><th>蜡烛</th>
  <th>收盘</th><th>RSI6</th><th>J</th><th>%B</th>
  <th>未入选原因</th><th>触发信号</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>

<h2>K线详细图（通过波动率过滤 Top 10，含布林带·RSI·KDJ，可展开）</h2>
<p style="font-size:12px;color:#666">点击展开查看K线图。布林带上轨(橙)、中轨(蓝)、下轨(橙)叠加于K线图。</p>
{charts_section}

</body>
</html>"""

out_path = "output/report_20260430_enhanced.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n[OK] Enhanced report saved: {out_path}")
print(f"  File size: {os.path.getsize(out_path):,} bytes")
print(f"\nFull path: {os.path.abspath(out_path)}")
