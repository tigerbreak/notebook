"""
员工查询工具
对应 RAGFlow: agent/tools/tavily.py 类似的结构
"""

from typing import Any, Dict, Optional

from .base import BaseTool


class GetEmployees(BaseTool):
    """查询员工信息 API 工具"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url.rstrip("/")
    
    @property
    def name(self) -> str:
        return "get_employees"
    
    @property
    def description(self) -> str:
        return (
            "查询公司员工信息列表。可按部门筛选员工，"
            "返回员工姓名、部门、职位等信息。"
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "可选，按部门筛选",
                    "enum": ["工程部", "产品部", "市场部", "人事部"]
                }
            },
            "required": []
        }
    
    async def __call__(self, department: Optional[str] = None, **kwargs) -> str:
        import httpx
        
        params = {}
        if department:
            params["department"] = department
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.api_base_url}/api/employees", params=params)
            data = resp.json()
        
        employees = data["employees"]
        if not employees:
            return f"未找到{department or '所有'}部门的员工"
        
        lines = [f"共找到 {data['total']} 名员工:"]
        for emp in employees:
            lines.append(f"  - {emp['name']} | {emp['department']} | {emp['position']}")
        return "\n".join(lines)
