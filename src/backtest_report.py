"""
Generate backtest HTML report.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _metrics_table_html(results: dict, hold_periods: list[int]) -> str:
    rows = ""
    for h in hold_periods:
        m = results.get(f"metrics_{h}", {})
        if not m:
            continue
        wr = m.get("win_rate", 0) or 0
        exc_wr = m.get("exc_win_rate", np.nan)
        exc_wr_str = f"{exc_wr:.1%}" if not (isinstance(exc_wr, float) and np.isnan(exc_wr)) else "N/A"
        exc_ret = m.get("mean_excess", np.nan)
        exc_ret_str = f"{exc_ret:.2%}" if not (isinstance(exc_ret, float) and np.isnan(exc_ret)) else "N/A"

        color = "#d4f0c0" if wr >= 0.55 else ("#fff3c0" if wr >= 0.50 else "#ffd4d4")
        rows += f"""<tr style="background:{color}">
  <td>{h}日</td>
  <td>{m.get('count',0)}</td>
  <td>{wr:.1%}</td>
  <td>{m.get('mean_return',0):.2%}</td>
  <td>{m.get('median_return',0):.2%}</td>
  <td>{m.get('max_loss',0):.2%}</td>
  <td>{exc_wr_str}</td>
  <td>{exc_ret_str}</td>
</tr>"""
    return f"""<table border="1" cellpadding="6" cellspacing="0"
style="border-collapse:collapse;font-size:13px">
<thead style="background:#2c3e50;color:white">
<tr><th>持有期</th><th>交易笔数</th><th>胜率</th><th>平均收益</th>
<th>中位收益</th><th>最大亏损</th><th>超额胜率</th><th>平均超额</th></tr>
</thead><tbody>{rows}</tbody></table>"""


def _monthly_heatmap_html(heatmap_df: pd.DataFrame) -> str:
    if heatmap_df.empty:
        return "<p>无数据</p>"
    months = ["1月","2月","3月","4月","5月","6月",
              "7月","8月","9月","10月","11月","12月"]
    years = sorted(heatmap_df.columns.tolist())
    header = "<th>月份</th>" + "".join(f"<th>{y}</th>" for y in years)
    rows = ""
    for m in range(1, 13):
        cells = f"<td><b>{months[m-1]}</b></td>"
        for y in years:
            val = heatmap_df.loc[m, y] if m in heatmap_df.index and y in heatmap_df.columns else np.nan
            if isinstance(val, float) and np.isnan(val):
                cells += "<td>-</td>"
            else:
                color = "#c8f0c8" if val > 0.02 else ("#f0f0c8" if val > 0 else "#f0c8c8")
                cells += f'<td style="background:{color}">{val:.1%}</td>'
        rows += f"<tr>{cells}</tr>"
    return f"""<table border="1" cellpadding="4" cellspacing="0"
style="border-collapse:collapse;font-size:12px">
<thead style="background:#2c3e50;color:white"><tr>{header}</tr></thead>
<tbody>{rows}</tbody></table>"""


def save_backtest_html(results: dict,
                       hold_periods: list[int],
                       failed_criteria: list[str],
                       passed: bool,
                       auto_tuned: bool,
                       output_path: str,
                       start_date: str,
                       end_date: str):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    status_color = "#d4f0c0" if passed else "#ffd4d4"
    status_text = "✅ 策略通过验证" if passed else "❌ 策略未通过验证（见诊断）"
    tuned_note = "（已自动调参）" if auto_tuned else "（原始参数）"

    criteria_html = ""
    if failed_criteria:
        items = "".join(f"<li>{c}</li>" for c in failed_criteria)
        criteria_html = f"<p><b>未达标项：</b></p><ul>{items}</ul>"
    else:
        criteria_html = "<p>所有指标均达标。</p>"

    metrics_html = _metrics_table_html(results, hold_periods)

    heatmap_sections = ""
    for h in hold_periods:
        from src.backtest import build_monthly_heatmap
        df_t = results.get(f"trades_{h}", pd.DataFrame())
        hm = build_monthly_heatmap(df_t)
        heatmap_sections += f"<h3>{h}日持有 — 月度平均收益热力图</h3>{_monthly_heatmap_html(hm)}"

    # sample trades table (last 50)
    trades_html = ""
    for h in hold_periods:
        df_t = results.get(f"trades_{h}", pd.DataFrame())
        if df_t.empty:
            continue
        tail = df_t.tail(50)
        cols = ["signal_date","buy_date","sell_date","code","buy_price",
                "sell_price","return","bench_return","excess_return","score","signals"]
        cols = [c for c in cols if c in tail.columns]
        header = "".join(f"<th>{c}</th>" for c in cols)
        rows = ""
        for _, row in tail.iterrows():
            color = "#d4f0c0" if row.get("return", 0) > 0 else "#ffd4d4"
            cells = "".join(f"<td>{row[c]}</td>" for c in cols)
            rows += f'<tr style="background:{color}">{cells}</tr>'
        trades_html += f"""<h3>{h}日持有 — 最近50笔交易记录</h3>
<div style="overflow-x:auto">
<table border="1" cellpadding="4" cellspacing="0"
style="border-collapse:collapse;font-size:11px;white-space:nowrap">
<thead style="background:#2c3e50;color:white"><tr>{header}</tr></thead>
<tbody>{rows}</tbody></table></div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>A股主板反转策略回测报告</title>
<style>
body{{font-family:Microsoft YaHei,sans-serif;margin:20px;background:#f9f9f9}}
h1{{color:#2c3e50}} h2,h3{{color:#34495e;margin-top:28px}}
.status{{padding:12px;border-radius:6px;margin:12px 0;font-size:14px;font-weight:bold}}
.warn{{background:#fff3c0;padding:12px;border-left:4px solid #f0a500;margin:12px 0}}
</style>
</head>
<body>
<h1>A股主板反转策略 — 3年回测报告 {tuned_note}</h1>
<p>回测区间：{start_date} ~ {end_date} &nbsp;|&nbsp;
   生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="status" style="background:{status_color}">{status_text}</div>
{criteria_html}

{'<div class="warn"><b>注意：</b>调参后参数已偏离原始设定，请在实盘前充分评估过拟合风险。</div>' if auto_tuned else ''}
{'<div class="warn"><b>重要提示：</b>策略在当前3年样本下未通过验证标准，不建议实盘使用。请查看诊断详情。</div>' if not passed and not auto_tuned else ''}

<h2>绩效汇总</h2>
{metrics_html}

<h2>月度收益热力图</h2>
{heatmap_sections}

<h2>交易明细（最近50笔）</h2>
{trades_html}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Backtest HTML saved: {output_path}")
