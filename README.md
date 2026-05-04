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
| 30日最大振幅 | ≥ 15% | 曾有弹性 |
| ATR% 近5日最大 | > 90日均值×1.2 | 活跃度验证 |
| BB带宽 | 近期收窄中 | 整理压缩形态 |

**第二层 — 五维共振评分**（满分 15 分，入选 ≥ 9 分）

| 维度 | 分值 | 核心信号 |
|------|------|---------|
| BOLL | 0–3 | 近3日触及下轨（必要条件）+ 带宽收窄 + %B连续回升 |
| RSI | 0–3 | RSI6 < 40 + 底背离 + 连续2日回升 |
| KDJ | 0–3 | K上穿D金叉 + J向上 + 低位（K<30） |
| 量价 | 0–3 | 路径A缩量止跌 / 路径B放量阳线确认 / 路径C单日缩量 / 路径D温和缩量 |
| 蜡烛 | 0–3 | 金针探底（创10日新低+长下影+高位收盘）/ 锤子线 / 强势阳线 |

**入选三重条件：**
1. BOLL ≥ 1（硬性，不满足直接淘汰）
2. 量价止跌确认：`vol≥2` 或 `candle≥3` 或 `(vol≥1 且 candle≥2)`
3. 总分 ≥ 9

### 回测结果（2021–2026，57笔）

| 指标 | 数值 |
|------|------|
| 胜率 | 52.6% |
| 止盈（+10%或布林上轨） | 52.6% |
| 止损（-10%） | 45.6% |
| 平均收益 | -0.48% |
| 中位数收益 | 1.44% |
| 最佳单笔 | +14.12% |
| 最差单笔 | -10.37% |
| 平均持有天数 | 17.1 天 |

> **注意**：2022年熊市对策略影响显著（当年止损率约63%）。建议配合市场环境过滤使用。

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
# 今日选股
python run.py pick

# 指定日期选股
python run.py pick --date 2026-04-24

# 指定输出数量
python run.py pick --top 20
```

### 4. 运行回测

```bash
# VectorBT向量化回测（推荐，速度快10-20x）
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
  amp30_min: 0.15          # 30日最大振幅门槛
  atr_pct_ratio: 1.2       # ATR扩张倍数
  min_close: 3.0           # 最低价格
  min_avg_amount: 50000000 # 最低成交额（5000万）

scoring:
  threshold: 9             # 入选总分门槛（满分15）
  rsi_long_oversold: 40    # RSI超卖线

backtest:
  start_date: "2021-01-04"
  end_date: "2026-04-25"
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
