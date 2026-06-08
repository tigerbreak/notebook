
"""员工查询工具 — 调用 /api/employees 获取员工信息"""
from typing import Any, Dict, Optional
import httpx
from .base import BaseTool


# 模拟员工数据（服务器不可用时使用）
MOCK_EMPLOYEES = {
    "all": [
        {"name": "张三", "department": "工程部", "position": "高级工程师"},
        {"name": "李四", "department": "工程部", "position": "前端开发"},
        {"name": "王五", "department": "产品部", "position": "产品经理"},
        {"name": "赵六", "department": "市场部", "position": "市场总监"},
        {"name": "钱七", "department": "人事部", "position": "HRBP"},
    ]
}


class GetEmployees(BaseTool):
    """查询公司员工信息"""

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
                    "enum": ["工程部", "产品部", "市场部", "人事部"],
                }
            },
            "required": [],
        }

    async def __call__(self, department: Optional[str] = None, **kwargs) -> str:
        params = {}
        if department:
            params["department"] = department

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base_url}/api/employees",
                    params=params,
                    timeout=5.0,
                )
                data = resp.json()
        except Exception:
            return self._mock_response(department)

        employees = data.get("employees", [])
        if not employees:
            return f"未找到{department or '所有'}部门的员工"

        lines = [f"共找到 {data.get('total', len(employees))} 名员工:"]
        for emp in employees:
            lines.append(f"  - {emp['name']} | {emp['department']} | {emp['position']}")
        return "\n".join(lines)

    def _mock_response(self, department: Optional[str] = None) -> str:
        """服务器不可用时返回模拟数据"""
        if department:
            data = [e for e in MOCK_EMPLOYEES["all"] if e["department"] == department]
        else:
            data = MOCK_EMPLOYEES["all"]
        if not data:
            return f"未找到{department or '所有'}部门的员工"
        lines = [f"共找到 {len(data)} 名员工（模拟数据）:"]
        for emp in data:
            lines.append(f"  - {emp['name']} | {emp['department']} | {emp['position']}")
        return "\n".join(lines)
