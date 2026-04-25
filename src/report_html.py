"""
HTML report generator using pyecharts.
Produces:
  - picks report (report_YYYYMMDD.html) with ranked table + per-stock charts
  - validate report (validate_CODE.html) for single-stock indicator verification
"""
import os
from datetime import datetime

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Candlestick, Grid, Line, Bar, Page
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType


def _kline_chart(df: pd.DataFrame, title: str = "", n_bars: int = 60) -> Grid:
    df = df.tail(n_bars).copy()
    dates = df["date"].tolist()
    ohlc = df[["open", "close", "low", "high"]].astype(float).values.tolist()
    volume = df["volume"].astype(float).tolist()
    vol_colors = ["#ef232a" if row[1] >= row[0] else "#14b143"
                  for row in ohlc]

    # Candlestick
    candle = (
        Candlestick()
        .add_xaxis(dates)
        .add_yaxis(
            "K线",
            ohlc,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a", color0="#14b143",
                border_color="#ef232a", border_color0="#14b143",
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(is_scale=True, axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(is_scale=True, splitarea_opts=opts.SplitAreaOpts(is_show=True)),
            datazoom_opts=[
                opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0, 1, 2, 3], range_start=0, range_end=100),
                opts.DataZoomOpts(is_show=True, xaxis_index=[0, 1, 2, 3], range_start=0, range_end=100),
            ],
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    # Boll overlay
    if "bb_upper" in df.columns:
        for band_col, color, name in [("bb_upper", "#ff9900", "BB上轨"),
                                       ("bb_mid", "#3399ff", "BB中轨"),
                                       ("bb_lower", "#ff9900", "BB下轨")]:
            vals = df[band_col].astype(float).round(3).tolist()
            candle.overlap(
                Line()
                .add_xaxis(dates)
                .add_yaxis(name, vals, is_symbol_show=False,
                           linestyle_opts=opts.LineStyleOpts(width=1, color=color, type_="dashed"),
                           label_opts=opts.LabelOpts(is_show=False))
            )

    # Volume bar
    vol_bar = (
        Bar()
        .add_xaxis(dates)
        .add_yaxis("成交量", volume,
                   itemstyle_opts=opts.ItemStyleOpts(
                       color=JsCode(
                           "function(params){var c=['#ef232a','#14b143'];"
                           "return c[params.dataIndex%2==0?0:1];}"
                       )
                   ),
                   label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}")),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    # RSI
    rsi_line = Line().add_xaxis(dates)
    for i, (col, color) in enumerate([("rsi6", "#ff4444"), ("rsi12", "#ff9900"), ("rsi24", "#3399ff")]):
        if col in df.columns:
            markline = None
            if i == 0:
                markline = opts.MarkLineOpts(data=[
                    opts.MarkLineItem(y=20, name="超卖",
                                      linestyle_opts=opts.LineStyleOpts(color="#aaa", type_="dashed")),
                    opts.MarkLineItem(y=80, name="超买",
                                      linestyle_opts=opts.LineStyleOpts(color="#aaa", type_="dashed")),
                ])
            rsi_line.add_yaxis(col.upper(), df[col].astype(float).round(2).tolist(),
                               is_symbol_show=False,
                               linestyle_opts=opts.LineStyleOpts(width=1.5, color=color),
                               label_opts=opts.LabelOpts(is_show=False),
                               markline_opts=markline)
    rsi_line.set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(min_=0, max_=100),
        legend_opts=opts.LegendOpts(pos_top="0%"),
    )

    # KDJ
    kdj_line = Line().add_xaxis(dates)
    for col, color in [("K", "#ff4444"), ("D", "#3399ff"), ("J", "#888888")]:
        if col in df.columns:
            kdj_line.add_yaxis(col, df[col].astype(float).round(2).tolist(),
                               is_symbol_show=False,
                               linestyle_opts=opts.LineStyleOpts(width=1.5, color=color),
                               label_opts=opts.LabelOpts(is_show=False))
    kdj_line.set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
        legend_opts=opts.LegendOpts(pos_top="0%"),
    )

    grid = (
        Grid(init_opts=opts.InitOpts(width="100%", height="620px", theme=ThemeType.WHITE))
        .add(candle, grid_opts=opts.GridOpts(pos_left="5%", pos_right="2%", pos_top="8%", height="42%"))
        .add(vol_bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="2%", pos_top="54%", height="8%"))
        .add(rsi_line, grid_opts=opts.GridOpts(pos_left="5%", pos_right="2%", pos_top="66%", height="12%"))
        .add(kdj_line, grid_opts=opts.GridOpts(pos_left="5%", pos_right="2%", pos_top="82%", height="12%"))
    )
    return grid


def _score_table_html(picks_df: pd.DataFrame) -> str:
    cols = ["rank", "code", "name", "industry", "total_score",
            "rsi_score", "kdj_score", "boll_score", "vol_score",
            "close", "rsi6", "J", "bb_width", "amp30", "signals"]
    cols = [c for c in cols if c in picks_df.columns]
    rows_html = ""
    for _, row in picks_df.iterrows():
        score = int(row.get("total_score", 0))
        color = "#d4f0c0" if score >= 85 else ("#fff3c0" if score >= 70 else "#ffd4d4")
        cells = "".join(f"<td>{row[c]}</td>" for c in cols)
        rows_html += f'<tr style="background:{color}">{cells}</tr>\n'
    headers = "".join(f"<th>{c}</th>" for c in cols)
    return f"""
<table border="1" cellpadding="4" cellspacing="0"
       style="border-collapse:collapse;font-size:12px;width:100%">
<thead style="background:#2c3e50;color:white"><tr>{headers}</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""


def save_picks_html(picks: list[dict],
                    stock_data: dict[str, pd.DataFrame],
                    meta: dict[str, dict],
                    picks_df: pd.DataFrame,
                    output_path: str,
                    scan_date: str,
                    config_summary: str = "",
                    backtest_summary: str = ""):
    """Generate full picks HTML report."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Build per-stock charts
    chart_htmls = []
    for p in picks[:30]:  # limit charts to top 30
        code = p["code"]
        m = meta.get(code, {})
        name = m.get("name", code)
        df = stock_data.get(code)
        if df is None or len(df) < 14:
            continue
        chart = _kline_chart(df, title=f"{name}({code})")
        chart_htmls.append((code, name, p["total_score"],
                             ";".join(p.get("signals", [])),
                             chart.render_embed()))

    table_html = _score_table_html(picks_df)

    charts_section = ""
    for code, name, score, signals, embed in chart_htmls:
        charts_section += f"""
<details style="margin:8px 0;border:1px solid #ccc;border-radius:4px">
  <summary style="padding:8px;cursor:pointer;background:#f5f5f5;font-weight:bold">
    [{int(score)}分] {name} ({code}) &nbsp;&nbsp;
    <span style="color:#666;font-size:11px">{signals}</span>
  </summary>
  <div style="padding:8px">{embed}</div>
</details>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>A股主板反转选股 {scan_date}</title>
<style>
body{{font-family:Microsoft YaHei,sans-serif;margin:20px;background:#f9f9f9}}
h1{{color:#2c3e50}} h2{{color:#34495e;margin-top:30px}}
.summary{{background:#eaf4fb;padding:12px;border-radius:6px;margin:10px 0}}
.config{{background:#f0f0f0;padding:8px;font-size:11px;font-family:monospace;white-space:pre-wrap}}
</style>
</head>
<body>
<h1>A股主板RSI+KDJ+布林三指标共振反转策略</h1>
<div class="summary">
  <b>扫描日期：</b>{scan_date} &nbsp;|&nbsp;
  <b>命中数量：</b>{len(picks)} 只 &nbsp;|&nbsp;
  <b>生成时间：</b>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
{('<div class="summary"><b>回测历史表现：</b>' + backtest_summary + '</div>') if backtest_summary else ''}
{('<div class="config">' + config_summary + '</div>') if config_summary else ''}

<h2>选股结果汇总表</h2>
{table_html}

<h2>K线 + 指标图（Top 30）</h2>
{charts_section}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved: {output_path}")


def save_validate_html(df: pd.DataFrame, code: str, output_path: str):
    """Single-stock validation HTML with last 20 bars indicator table + charts."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    chart = _kline_chart(df, title=f"指标验证 {code}", n_bars=120)

    # last 20 rows table
    indicator_cols = ["date", "close", "rsi6", "rsi12", "rsi24",
                      "K", "D", "J", "bb_upper", "bb_mid", "bb_lower", "bb_width",
                      "atr_pct", "amp30"]
    indicator_cols = [c for c in indicator_cols if c in df.columns]
    tail = df.tail(20)[indicator_cols].round(4)
    rows_html = ""
    for _, row in tail.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in indicator_cols)
        rows_html += f"<tr>{cells}</tr>\n"
    headers = "".join(f"<th>{c}</th>" for c in indicator_cols)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>指标验证 {code}</title>
<style>
body{{font-family:Microsoft YaHei,sans-serif;margin:20px}}
table{{border-collapse:collapse;font-size:11px}}
th,td{{border:1px solid #ccc;padding:4px 8px}}
th{{background:#2c3e50;color:white}}
</style>
</head>
<body>
<h1>单股指标验证: {code}</h1>
<p>用以下数值与同花顺/通达信核对。RSI/Boll误差 &lt; 0.5% 视为通过；
KDJ前10日因初值差异属正常。</p>

<h2>近20个交易日指标值</h2>
<table><thead><tr>{headers}</tr></thead>
<tbody>{rows_html}</tbody></table>

<h2>K线 + RSI + KDJ + 布林带（近120日）</h2>
{chart.render_embed()}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Validate HTML saved: {output_path}")
