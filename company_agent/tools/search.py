
"""内部搜索工具 — 调用 /api/search"""
from typing import Any, Dict
import httpx
from .base import BaseTool


# 模拟搜索数据（服务器不可用时使用）
MOCK_SEARCH = {
    "AI": [
        {"title": "AI 战略规划文档", "snippet": "公司将在 Q4 启动 AI 中台建设，重点投入 NLP 和推荐系统方向..."},
        {"title": "智能客服项目立项", "snippet": "基于大语言模型的智能客服项目已获批准，预计年底上线..."},
    ],
    "默认": [
        {"title": "公司管理制度 V3.2", "snippet": "本制度适用于全体员工，涵盖考勤、福利、绩效考核等方面..."},
        {"title": "季度 OKR 指引", "snippet": "请各部门在每月初提交本季度 OKR，由管理层评审后发布..."},
    ],
}


class SearchCompany(BaseTool):
    """在公司内部知识库中搜索"""

    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "search_company"

    @property
    def description(self) -> str:
        return (
            "在公司内部知识库中搜索信息。当无法通过结构化 API "
            "获取信息时使用此工具。返回相关文档摘要。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        }

    async def __call__(self, query: str, **kwargs) -> str:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base_url}/api/search",
                    params={"q": query},
                    timeout=5.0,
                )
                data = resp.json()
        except Exception:
            return self._mock_response(query)

        results = data.get("results", [])
        if not results:
            return f"未找到关于 '{query}' 的信息"

        lines = [f"找到 {len(results)} 条关于 '{query}' 的结果:"]
        for r in results:
            lines.append(
                f"  - {r.get('title', '无标题')}: "
                f"{r.get('snippet', '')[:100]}"
            )
        return "\n".join(lines)

    def _mock_response(self, query: str) -> str:
        """服务器不可用时返回模拟数据"""
        # 尝试匹配关键词
        for keyword, results in MOCK_SEARCH.items():
            if keyword != "默认" and keyword.lower() in query.lower():
                break
        else:
            results = MOCK_SEARCH["默认"]
        lines = [f"找到 {len(results)} 条关于 '{query}' 的结果（模拟数据）:"]
        for r in results:
            lines.append(f"  - {r.get('title', '无标题')}: {r.get('snippet', '')[:100]}")
        return "\n".join(lines)
