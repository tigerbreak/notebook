"""
公司指标查询工具
"""

from typing import Any, Dict

from .base import BaseTool


class GetMetrics(BaseTool):
    """获取公司关键业务指标 API 工具"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url.rstrip("/")
    
    @property
    def name(self) -> str:
        return "get_metrics"
    
    @property
    def description(self) -> str:
        return (
            "获取公司关键业务指标，包括营收、活跃用户数、"
            "用户满意度、系统可用性等数据。"
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def __call__(self, **kwargs) -> str:
        import httpx
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.api_base_url}/api/metrics")
            data = resp.json()
        
        rev = data["revenue"]
        return (
            f"Q3 营收: {rev['q3']:,} {rev['unit']} | Q4 目标: {rev['q4_target']:,} {rev['unit']}\n"
            f"活跃用户: {data['users']['active']:,} | 本月新增: {data['users']['new_this_month']:,}\n"
            f"用户满意度: {data['satisfaction']}/5.0\n"
            f"系统可用率: {data['uptime']}%"
        )
