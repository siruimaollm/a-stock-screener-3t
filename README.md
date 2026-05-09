# A股主板超卖反弹选股器

基于 **RSI + KDJ + 布林带三指标共振** 的 A 股主板候选股筛选系统，配合 VectorBT 向量化回测与东方财富妙想 API 数据增强。

---

## 策略逻辑

### 选股思路

筛选「曾剧烈波动、近期布林带收窄整理、触及下轨后出现超卖反弹共振」的主板标的。

### 两层过滤

**第一层 — 波动率门槛**（全部 AND）

| 条件 | 默认值 | 说明 |
|------|--------|------|
| 收盘价 | ≥ 3 元 | 剔除退市风险 |
| 5日均成交额 | ≥ 5000 万 | 流动性硬门槛 |
| 历史窗口涨幅 | ≥ 25% | 过去 60 日内曾被资金推升 |
| 上涨段均量 | > 90日均量×1.3 | 上涨过程必须放量 |
| 上涨持续天数 | ≥ 5 日 | 避免单日脉冲误判 |
| 30日价格区间/均价 | ≤ 20% | 近期整理收敛 |
| 当前价 / 60日高点 | 40% ~ 90% | 回撤受控，不能太高也不能坍塌 |
| 10日均量 vs 60日均量 | 10日 < 60日 | 只保留方向性缩量，避免与第二层重复惩罚 |
| 10日ATR vs 30日ATR | 10日 < 30日×0.85 | 波动率收敛 |
| 近5日累计涨跌幅 | > -10% | 避开仍在急跌的标的 |

**第二层 — 五维共振评分**（满分 15 分，分层入选）

| 维度 | 分值 | 核心信号 |
|------|------|---------|
| BOLL | 0–3 | 近3日触及下轨（必要条件）+ 带宽收窄 + %B连续回升 |
| RSI | 0–3 | RSI6 < 40 + 底背离 + 连续2日回升 |
| KDJ | 0–3 | K上穿D金叉 + J向上 + 低位（K<30） |
| 量价 | 0–3 | 路径A缩量止跌 / 路径B放量阳线确认 / 路径C单日缩量 / 路径D温和缩量 |
| 蜡烛 | 0–3 | 金针探底（创10日新低+长下影+高位收盘）/ 锤子线 / 强势阳线 |

**入选三重条件：**
1. BOLL ≥ 1（硬性，不满足直接淘汰）
2. 量价/蜡烛确认分层：
   - 强确认：`vol≥2` 或 `candle≥3` 或 `(vol≥1 且 candle≥2)`
   - 弱确认：`candle≥2` 或 `(vol≥1 且 candle≥1)`
   - 无确认：直接淘汰
3. 分层总分门槛：
   - 强确认：总分 ≥ 8
   - 弱确认：总分 ≥ 9
4. 附加入场质量约束：
   - `candle_score ≥ 1`
   - `rsi_score + kdj_score ≥ 1`

### 交易与回测规则（当前默认）

- 回测模式：`backtest-v2`
- 信号频率：半月一次（`biweekly`）
- 买入：信号次日开盘
- 止盈：`+10%` 或 `收盘触及布林上轨`
- 止损：日内触及入场价 `-8%` 即止损
- 失效退出：持有 `8` 个交易日仍未转正则退出
- 最长持有：`252` 个交易日

### 最新回测结果（v2，2021-01-04 ~ 2026-05-07）

| 指标 | 数值 |
|------|------|
| 总交易笔数 | 54 |
| 胜率 | 40.7% |
| 止盈率（+10%或布林上轨） | 40.7% |
| 破位止损率（-8%） | 7.4% |
| 失效退出率（8日未转正） | 51.9% |
| 平均收益 | 1.62% |
| 中位数收益 | -0.40% |
| 平均超额收益（vs 沪深300） | 1.44% |
| 超额胜率（vs 沪深300） | 53.7% |
| 最佳单笔 | +10.00% |
| 最差单笔 | -8.00% |
| 平均持有天数 | 7.7 天 |

> 数据口径：选股/回测结束日默认取 `stock_data.db` 中最新可用交易日；上表对应当前默认参数的一次实际回测运行结果。
> 这版是“更保守、先控回撤”的默认方案，牺牲了部分胜率，换来更短持有期和更可控的最差单笔。

---

## 项目结构

```
ashare_screener/
├── run.py                    # 主入口（CLI）
├── config.yaml               # 所有参数集中配置
├── requirements.txt
├── src/
│   ├── indicators.py         # 技术指标计算（numba加速KDJ）
│   ├── volatility_filter.py  # 第一层：波动率过滤
│   ├── scoring.py            # 第二层：五维评分
│   ├── data_fetcher.py       # 数据加载（SQLite / DuckDB）
│   ├── backtest_vbt.py       # VectorBT向量化回测
│   ├── backtest_tp.py        # 简单止盈止损回测
│   ├── universe.py           # 股票池（主板过滤）
│   ├── report_html.py        # HTML报告生成
│   ├── report_csv.py         # CSV输出
│   └── miao_api.py           # 东方财富妙想API封装
├── data/                     # 行情数据库（本地，不入库）
│   └── stock_data.db         # SQLite，5年后复权日线
└── output/                   # 选股结果输出（本地生成）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install vectorbt numba
```

### 2. 准备数据

将 `stock_data.db`（SQLite，含 `daily_data_hfq` 表）放入 `data/` 目录。

字段要求：`stock_code, date, open, high, low, close, volume, amount, turnover_rate`

### 3. 运行选股

```bash
# 使用数据库中最新交易日选股
python run.py pick

# 指定日期选股
python run.py pick --date 2026-04-24

# 指定输出数量
python run.py pick --top 20
```

### 4. 运行回测

```bash
# v2 回测（当前默认参考口径）
python run.py backtest-v2

# VectorBT 向量化回测
python run.py backtest-vbt

# 简单止盈止损回测
python run.py backtest-tp
```

### 5. 妙想 API（可选）

配置 API Key：

```bash
# Windows
$env:MX_APIKEY = "your_api_key"

# Linux / macOS
export MX_APIKEY=your_api_key
```

使用：

```bash
# 查询金融数据
python run.py miao-data --query "贵州茅台近一年收盘价"

# 搜索资讯研报
python run.py miao-search --query "宁德时代最新研报"

# 智能选股
python run.py miao-xuangu --query "今日A股主板涨停股票"
```

---

## 配置说明（config.yaml）

```yaml
volatility:
  rally_min_pct: 0.25      # 历史窗口最小涨幅
  directional_vol_shrink: true
  vol_shrink_short_window: 10
  vol_shrink_long_window: 60
  min_close: 3.0           # 最低价格
  min_avg_amount: 50000000 # 最低成交额（5000万）

scoring:
  threshold: 8             # 强确认总分门槛（满分15）
  weak_confirmation_threshold: 9
  rsi_long_oversold: 40    # RSI超卖线
  min_candle_score: 1
  require_rsi_or_kdj_score_sum: 1

backtest:
  start_date: "2021-01-04"
  end_date: "2026-05-07"

backtest_v2:
  take_profit: 0.10
  break_down_pct: 0.08
  stop_loss_mode: "intraday"
  fail_exit_days: 8
  fail_exit_min_return: 0.0
  max_hold: 252
  signal_freq: "biweekly"
```

---

## 技术栈

| 组件 | 用途 |
|------|------|
| pandas / numpy | 数据处理 |
| numba | KDJ指标JIT加速（~50×） |
| VectorBT | 向量化回测（全市场并行） |
| SQLite | 5年后复权行情存储 |
| DuckDB | 原始行情查询（兼容旧版） |
| Jinja2 | HTML报告模板渲染 |
| 东方财富妙想API | 实时行情 / 资讯 / 智能选股 |

---

## 免责声明

本项目仅供学习研究，不构成任何投资建议。股市有风险，投资需谨慎。
