"""
妙想 API 客户端封装（东方财富）
=================================
封装三个接口：
  MiaoData    — 金融数据查询  /finskillshub/api/claw/query
  MiaoSearch  — 资讯/研报搜索  /finskillshub/api/claw/news-search
  MiaoXuangu  — 智能选股       /finskillshub/api/claw/stock-screen

配置 API Key:
  方式1（推荐）：设置环境变量
      Windows: $env:MX_APIKEY = "your_key"
      Linux:   export MX_APIKEY=your_key
  方式2：config.yaml 中 miao.api_key
  方式3：实例化时直接传入 api_key 参数

基本用法:
  from src.miao_api import MiaoData, MiaoSearch, MiaoXuangu

  data = MiaoData().query("贵州茅台近一年收盘价")
  news = MiaoSearch().search("宁德时代最新研报")
  picks = MiaoXuangu().screen("今日涨跌幅大于2%且市盈率小于30的A股主板股票")
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

# ─────────────────────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────────────────────

_BASE = "https://mkapi2.dfcfs.com/finskillshub/api/claw"
_TIMEOUT = 30


def _get_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.getenv("MX_APIKEY", "")
    if not key:
        raise ValueError(
            "未找到 MX_APIKEY。请：\n"
            "  Windows: $env:MX_APIKEY = 'your_key'\n"
            "  Linux:   export MX_APIKEY=your_key\n"
            "或在实例化时传入 api_key='your_key'"
        )
    return key


def _headers(api_key: str) -> Dict[str, str]:
    return {"Content-Type": "application/json", "apikey": api_key}


def _safe_name(s: str, max_len: int = 60) -> str:
    s = re.sub(r'[<>:"/\\|?*\[\]]', "_", s).strip().replace(" ", "_")
    return s[:max_len] or "query"


# ─────────────────────────────────────────────────────────────────────────────
#  MiaoData — 金融行情 / 财务数据
# ─────────────────────────────────────────────────────────────────────────────

class MiaoData:
    """
    妙想金融数据查询（行情、财务、股东、板块等）。

    支持自然语言问句，如：
      "贵州茅台近三年净利润"
      "宁德时代今日主力资金流向"
      "沪深300指数最新点位"
    """

    URL = f"{_BASE}/query"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = _get_api_key(api_key)

    def query(self, question: str) -> Dict[str, Any]:
        """发起查询，返回原始 JSON dict。"""
        resp = requests.post(
            self.URL,
            headers=_headers(self.api_key),
            json={"toolQuery": question},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def to_dataframes(result: Dict[str, Any]):
        """
        将原始 JSON 解析为 list[dict]，每项含 sheet_name + DataFrame。
        返回: (tables, error_msg)
          tables: [{"name": str, "df": pd.DataFrame}]
          error_msg: str | None
        """
        import pandas as pd

        if result.get("status") != 0:
            return [], f"API 错误 {result.get('status')}: {result.get('message')}"

        dto_list = (
            result
            .get("data", {})
            .get("data", {})
            .get("searchDataResultDTO", {})
            .get("dataTableDTOList", [])
        )
        if not dto_list:
            return [], "返回数据为空（dataTableDTOList 缺失）"

        tables = []
        for dto in dto_list:
            if not isinstance(dto, dict):
                continue
            table = dto.get("table", {})
            name_map = dto.get("nameMap", {})
            if isinstance(name_map, list):
                name_map = {str(i): v for i, v in enumerate(name_map)}
            headers = table.get("headName", []) if isinstance(table, dict) else []

            rows = []
            if headers and isinstance(table, dict):
                indicator_keys = [k for k in table if k != "headName"]
                for idx, date in enumerate(headers):
                    row = {"date": str(date)}
                    for k in indicator_keys:
                        label = str(name_map.get(k, name_map.get(str(k), k)))
                        vals = table[k]
                        row[label] = vals[idx] if isinstance(vals, list) and idx < len(vals) else ""
                    rows.append(row)

            if rows:
                sheet = _safe_name(dto.get("title") or dto.get("entityName") or "数据")
                tables.append({"name": sheet, "df": pd.DataFrame(rows)})

        if not tables:
            return [], "未解析出有效表格"
        return tables, None

    def query_df(self, question: str):
        """查询并直接返回第一个 DataFrame；无数据时返回空 DataFrame。"""
        import pandas as pd
        result = self.query(question)
        tables, err = self.to_dataframes(result)
        if err or not tables:
            print(f"[MiaoData] {err or '无数据'}")
            return pd.DataFrame()
        return tables[0]["df"]


# ─────────────────────────────────────────────────────────────────────────────
#  MiaoSearch — 资讯 / 研报 / 新闻搜索
# ─────────────────────────────────────────────────────────────────────────────

class MiaoSearch:
    """
    妙想资讯搜索（新闻、公告、研报、政策等）。

    支持自然语言问句，如：
      "贵州茅台最新研报"
      "今日大盘异动原因"
      "宁德时代定增预案解读"
    """

    URL = f"{_BASE}/news-search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = _get_api_key(api_key)

    def search(self, question: str) -> Dict[str, Any]:
        """发起搜索，返回原始 JSON dict。"""
        resp = requests.post(
            self.URL,
            headers=_headers(self.api_key),
            json={"query": question},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def to_text(result: Dict[str, Any]) -> str:
        """将原始 JSON 提取为可读文本。"""
        if result.get("status") != 0:
            return f"[错误] {result.get('status')}: {result.get('message')}"

        inner = result.get("data", {}).get("data", {})

        # 主路径：llmSearchResponse.data
        items = inner.get("llmSearchResponse", {}).get("data", [])

        # 兜底路径
        if not items:
            items = inner.get("results", []) or inner.get("content", [])
        if not items:
            return "未找到相关资讯"

        lines = []
        for i, item in enumerate(items, 1):
            title   = item.get("title", "")
            source  = item.get("mediaName", "") or item.get("source", "")
            date    = item.get("publishDate", "") or item.get("date", "")
            summary = item.get("summary", "") or item.get("content", "") or item.get("trunk", "")
            # 关联股票
            secus = item.get("secuList", [])
            secu_str = " ".join(f"{s.get('secuName','')}({s.get('secuCode','')})" for s in secus[:3]) if secus else ""

            lines.append(f"[{i}] {title}")
            if source or date:
                lines.append(f"    来源: {source}  日期: {date}  {secu_str}")
            if summary:
                lines.append("    " + summary[:250] + ("..." if len(summary) > 250 else ""))
            lines.append("")
        return "\n".join(lines)

    def search_text(self, question: str) -> str:
        """搜索并直接返回可读文本。"""
        return self.to_text(self.search(question))


# ─────────────────────────────────────────────────────────────────────────────
#  MiaoXuangu — 智能选股
# ─────────────────────────────────────────────────────────────────────────────

class MiaoXuangu:
    """
    妙想智能选股（按行情/财务条件筛选）。

    支持自然语言条件，如：
      "今日A股主板涨幅大于5%"
      "市盈率小于20且ROE大于15%"
      "半导体行业毛利率大于40%的公司"
    """

    URL = f"{_BASE}/stock-screen"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = _get_api_key(api_key)

    def screen(self, condition: str) -> Dict[str, Any]:
        """发起选股，返回原始 JSON dict。"""
        resp = requests.post(
            self.URL,
            headers=_headers(self.api_key),
            json={"keyword": condition},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def to_dataframe(result: Dict[str, Any]):
        """
        将原始 JSON 解析为 DataFrame。
        返回: (df, total, error_msg)
        """
        import pandas as pd

        if result.get("status") != 0:
            return pd.DataFrame(), 0, f"API 错误 {result.get('status')}: {result.get('message')}"

        inner = result.get("data", {}).get("data", {})
        all_res = inner.get("allResults", {}).get("result", {})
        data_list = all_res.get("dataList", [])
        columns = all_res.get("columns", [])
        total = all_res.get("total", 0)

        if data_list:
            # 构建列名映射
            col_map = {}
            col_order = []
            for col in columns:
                key = col.get("field") or col.get("key") or col.get("name", "")
                title = col.get("title") or col.get("displayName") or key
                date_msg = col.get("dateMsg", "")
                if date_msg:
                    title = f"{title} {date_msg}"
                if key:
                    col_map[str(key)] = str(title)
                    col_order.append(str(key))

            rows = []
            for row in data_list:
                new_row = {}
                for k in col_order:
                    label = col_map.get(k, k)
                    val = row.get(k, "")
                    new_row[label] = "" if val is None else str(val)
                # 补上未在 columns 的字段
                for k, v in row.items():
                    label = col_map.get(str(k), str(k))
                    if label not in new_row:
                        new_row[label] = "" if v is None else str(v)
                rows.append(new_row)

            return pd.DataFrame(rows), total, None

        # 回退：partialResults Markdown 表格
        partial = inner.get("partialResults", "")
        if partial:
            lines = [l.strip() for l in partial.strip().splitlines() if l.strip()]
            if lines:
                def split_cells(line):
                    return [c.strip() for c in line.split("|") if c.strip()]
                headers = split_cells(lines[0])
                data_start = 2 if len(lines) > 1 and re.match(r"^[\s|\-]+$", lines[1]) else 1
                rows = []
                for line in lines[data_start:]:
                    cells = split_cells(line)
                    if len(cells) < len(headers):
                        cells += [""] * (len(headers) - len(cells))
                    rows.append(dict(zip(headers, cells[:len(headers)])))
                return pd.DataFrame(rows), len(rows), None

        return pd.DataFrame(), 0, "返回数据为空"

    def screen_df(self, condition: str):
        """选股并直接返回 DataFrame。"""
        import pandas as pd
        result = self.screen(condition)
        df, total, err = self.to_dataframe(result)
        if err:
            print(f"[MiaoXuangu] {err}")
            return pd.DataFrame()
        print(f"[MiaoXuangu] 符合条件: {total} 只，返回 {len(df)} 行")
        return df
