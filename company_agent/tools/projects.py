"""
项目查询工具
"""

from typing import Any, Dict, Optional

from .base import BaseTool


class GetProjects(BaseTool):
    """查询项目信息 API 工具"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url.rstrip("/")
    
    @property
    def name(self) -> str:
        return "get_projects"
    
    @property
    def description(self) -> str:
        return (
            "查询公司项目列表。可按状态筛选项目，"
            "返回项目名称、状态、进度、负责团队等信息。"
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "可选，按状态筛选",
                    "enum": ["进行中", "规划中", "已完成"]
                }
            },
            "required": []
        }
    
    async def __call__(self, status: Optional[str] = None, **kwargs) -> str:
        import httpx
        
        params = {}
        if status:
            params["status"] = status
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.api_base_url}/api/projects", params=params)
            data = resp.json()
        
        projects = data["projects"]
        if not projects:
            return f"未找到状态为'{status or '所有'}的项目"
        
        lines = [f"共找到 {data['total']} 个项目:"]
        for proj in projects:
            lines.append(
                f"  - [{proj['id']}] {proj['name']} | {proj['status']} "
                f"| 进度{proj['progress']}% | {proj['team']}"
            )
        return "\n".join(lines)
