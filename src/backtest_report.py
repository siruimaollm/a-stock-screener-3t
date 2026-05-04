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


# ─────────────────────────────────────────────
#  v2 回测报告（TP=+10%/布林上轨, SL=破位, 1年超时）
# ─────────────────────────────────────────────

def save_backtest_v2_html(metrics: dict,
                           trades_df: pd.DataFrame,
                           output_path: str,
                           start_date: str,
                           end_date: str,
                           meta: dict = None):
    """Generate HTML report for backtest_v2 (TP/SL with break-down).

    meta: dict {code: {"name": ..., "industry": ...}} for stock name lookup.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n = metrics.get("total_trades", 0)
    if n == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"<h1>无交易</h1><p>区间 {start_date}~{end_date} 内策略未触发任何信号。</p>")
        print(f"Backtest v2 HTML saved (empty): {output_path}")
        return

    # 关键指标
    win = metrics["win_rate"]
    color_main = "#d4f0c0" if win >= 0.55 else ("#fff3c0" if win >= 0.50 else "#ffd4d4")

    # 月度热力图
    df = trades_df.copy()
    # 注入股票名称（meta 优先；否则尝试自动加载）
    if meta is None:
        try:
            from src.data_fetcher import load_stock_info_sqlite
            meta = load_stock_info_sqlite("data/stock_data.db")
        except Exception:
            meta = {}
    df["name"] = df["code"].map(lambda c: (meta.get(c, {}) or {}).get("name", "") if isinstance(meta, dict) else "")
    df["industry"] = df["code"].map(lambda c: (meta.get(c, {}) or {}).get("industry", "") if isinstance(meta, dict) else "")
    df["buy_dt"] = pd.to_datetime(df["buy_date"])
    df["year"] = df["buy_dt"].dt.year
    df["month"] = df["buy_dt"].dt.month
    pv_count = df.groupby(["year","month"]).size().unstack("year").fillna(0).astype(int)
    pv_ret   = df.groupby(["year","month"])["return"].mean().unstack("year").round(4)

    months = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
    years = sorted(pv_ret.columns.tolist()) if not pv_ret.empty else []
    heatmap_html = ""
    if years:
        header = "<th>月份</th>" + "".join(f"<th>{y}</th>" for y in years)
        rows = ""
        for m in range(1, 13):
            cells = f"<td>{months[m-1]}</td>"
            for y in years:
                v = pv_ret.get(y, pd.Series()).get(m, np.nan) if y in pv_ret.columns else np.nan
                c = pv_count.get(y, pd.Series()).get(m, 0) if y in pv_count.columns else 0
                if pd.isna(v):
                    cells += "<td style='color:#bbb'>-</td>"
                else:
                    if v > 0.03:    bg = "#7ed957"
                    elif v > 0:     bg = "#c8eba0"
                    elif v > -0.03: bg = "#ffd699"
                    else:           bg = "#ff8a8a"
                    cells += f"<td style='background:{bg}'>{v*100:+.1f}%<br><span style='font-size:10px;color:#444'>n={int(c)}</span></td>"
            rows += f"<tr>{cells}</tr>"
        heatmap_html = f"""<table style="border-collapse:collapse;font-size:12px">
<thead style="background:#2c3e50;color:white"><tr>{header}</tr></thead>
<tbody>{rows}</tbody></table>"""

    # 退出原因分布
    reason_counts = df["exit_reason"].value_counts()
    reason_rows = ""
    for r, c in reason_counts.items():
        sub = df[df["exit_reason"] == r]
        avg_ret  = sub["return"].mean()
        avg_hold = sub["hold_days"].mean()
        reason_rows += f"""<tr>
  <td>{r}</td><td>{c}</td><td>{c/n:.1%}</td>
  <td>{avg_ret:+.2%}</td><td>{avg_hold:.0f} 日</td>
</tr>"""

    # 个股交易统计（按出现次数排序，看哪些股票贡献最多）
    by_code = df.groupby(["code"]).agg(
        name=("name", "first"),
        industry=("industry", "first"),
        trades=("return", "count"),
        wins=("return", lambda x: (x > 0).sum()),
        mean_ret=("return", "mean"),
        sum_ret=("return", "sum"),
        max_ret=("return", "max"),
        min_ret=("return", "min"),
    ).reset_index()
    by_code["win_rate"] = by_code["wins"] / by_code["trades"]
    by_code = by_code.sort_values(["trades", "sum_ret"], ascending=[False, False]).head(30)

    by_code_rows = ""
    for _, row in by_code.iterrows():
        bg = "#d4f0c0" if row["mean_ret"] > 0.02 else ("#fff3c0" if row["mean_ret"] > 0 else "#ffd4d4")
        by_code_rows += f"""<tr style="background:{bg}">
  <td><b>{row['code']}</b></td>
  <td>{row['name']}</td>
  <td style="font-size:11px;color:#666">{row['industry']}</td>
  <td>{int(row['trades'])}</td>
  <td>{int(row['wins'])}/{int(row['trades'])}</td>
  <td>{row['win_rate']:.0%}</td>
  <td>{row['mean_ret']:+.2%}</td>
  <td>{row['max_ret']:+.2%}</td>
  <td>{row['min_ret']:+.2%}</td>
</tr>"""

    # 最近50笔（含股票名称）
    cols = ["signal_date","buy_date","exit_date","code","name","industry",
            "buy_price","exit_price","return","excess_return","hold_days","exit_reason","score","signals"]
    cols = [c for c in cols if c in df.columns]
    col_label = {
        "signal_date": "信号日", "buy_date": "买入日", "exit_date": "卖出日",
        "code": "代码", "name": "名称", "industry": "行业",
        "buy_price": "买入价", "exit_price": "卖出价",
        "return": "收益率", "excess_return": "超额", "hold_days": "持有",
        "exit_reason": "退出原因", "score": "总分", "signals": "触发信号",
    }
    tail = df.sort_values("buy_date", ascending=False).head(50)
    header = "".join(f"<th>{col_label.get(c, c)}</th>" for c in cols)
    body_rows = ""
    for _, row in tail.iterrows():
        bg = "#d4f0c0" if row.get("return", 0) > 0 else "#ffd4d4"
        cells_html = ""
        for c in cols:
            val = row[c]
            if c == "return" or c == "excess_return":
                if pd.notna(val):
                    cells_html += f"<td>{float(val):+.2%}</td>"
                else:
                    cells_html += "<td>-</td>"
            elif c == "hold_days":
                cells_html += f"<td>{int(val)}日</td>"
            else:
                cells_html += f"<td>{val}</td>"
        body_rows += f"<tr style='background:{bg}'>{cells_html}</tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>A股主板反转策略 - v2 回测报告</title>
<style>
body{{font-family:Microsoft YaHei,sans-serif;margin:20px;background:#f9f9f9;line-height:1.6}}
h1{{color:#2c3e50}} h2{{color:#34495e;margin-top:30px;border-bottom:2px solid #eee;padding-bottom:6px}}
.metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}}
.metric{{background:white;border:1px solid #e0e0e0;border-radius:6px;padding:14px;text-align:center}}
.metric .v{{font-size:22px;font-weight:bold;color:#2c3e50}}
.metric .l{{font-size:12px;color:#666;margin-top:4px}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0}}
th{{background:#2c3e50;color:white;padding:6px 10px}}
td{{padding:5px 8px;border:1px solid #e0e0e0;text-align:center}}
.rules{{background:#eaf4fb;padding:12px;border-radius:6px;border-left:4px solid #3498db}}
</style>
</head>
<body>
<h1>A股主板反转策略 — v2 回测报告</h1>
<p>区间 <b>{start_date}</b> ~ <b>{end_date}</b> &nbsp;|&nbsp;
   生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="rules">
<b>v2 出场规则：</b><br>
&bull; <b>止盈</b>：收益≥+{metrics['take_profit']*100:.0f}%（按当日 high 触发）或 收盘≥布林上轨（先到先得）<br>
&bull; <b>止损</b>：收盘 ≤ entry × (1 - {metrics['break_down_pct']*100:.0f}%) ⇒ 破位平仓<br>
&bull; <b>超时</b>：持有满 {metrics['max_hold']} 个交易日（≈1年）按收盘价平仓
</div>

<h2>核心指标</h2>
<div class="metric-grid">
  <div class="metric"><div class="v">{n}</div><div class="l">总交易笔数</div></div>
  <div class="metric" style="background:{color_main}"><div class="v">{win:.1%}</div><div class="l">胜率（盈利率）</div></div>
  <div class="metric"><div class="v">{metrics['mean_return']:+.2%}</div><div class="l">平均收益</div></div>
  <div class="metric"><div class="v">{metrics['median_return']:+.2%}</div><div class="l">收益中位数</div></div>
  <div class="metric"><div class="v">{metrics['max_return']:+.2%}</div><div class="l">最优单笔</div></div>
  <div class="metric"><div class="v">{metrics['min_return']:+.2%}</div><div class="l">最差单笔</div></div>
  <div class="metric"><div class="v">{metrics['mean_hold_days']:.0f} 日</div><div class="l">平均持有</div></div>
  <div class="metric"><div class="v">{metrics.get('exc_win_rate',0):.1%}</div><div class="l">超额胜率(vs HS300)</div></div>
</div>

<h2>退出原因分布</h2>
<table>
<thead><tr><th>原因</th><th>笔数</th><th>占比</th><th>平均收益</th><th>平均持有</th></tr></thead>
<tbody>{reason_rows}</tbody>
</table>

<h2>月度收益热力图（按入场月统计）</h2>
{heatmap_html}

<h2>个股交易统计（Top 30，按交易次数排序）</h2>
<table style="font-size:12px">
<thead><tr>
  <th>代码</th><th>名称</th><th>行业</th>
  <th>交易次数</th><th>盈/总</th><th>胜率</th>
  <th>平均收益</th><th>最优单笔</th><th>最差单笔</th>
</tr></thead>
<tbody>{by_code_rows}</tbody>
</table>

<h2>最近 50 笔交易记录</h2>
<div style="overflow-x:auto">
<table style="white-space:nowrap;font-size:11px">
<thead><tr>{header}</tr></thead>
<tbody>{body_rows}</tbody>
</table>
</div>

</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Backtest v2 HTML saved: {output_path}")
