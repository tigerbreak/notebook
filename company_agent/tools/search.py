"""
公司内部搜索工具
"""

from typing import Any, Dict

from .base import BaseTool


class SearchCompany(BaseTool):
    """公司内部知识库搜索 API 工具"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url.rstrip("/")
    
    @property
    def name(self) -> str:
        return "search_company"
    
    @property
    def description(self) -> str:
        return (
            "在公司内部知识库中搜索信息。适用于查询公司相关的具体信息，"
            "如项目进展、团队情况等。"
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，建议简短具体"
                }
            },
            "required": ["query"]
        }
    
    async def __call__(self, query: str, **kwargs) -> str:
        import httpx
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.api_base_url}/api/search?query={query}")
            data = resp.json()
        
        results = data["results"]
        if not results:
            return f"搜索 '{query}' 未找到结果"
        
        lines = [f"搜索 '{query}' 找到 {len(results)} 条结果:"]
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r}")
        return "\n".join(lines)
