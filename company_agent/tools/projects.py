
"""项目查询工具 — 调用 /api/projects 获取项目信息"""
from typing import Any, Dict, Optional
import httpx
from .base import BaseTool


# 模拟项目数据（服务器不可用时使用）
MOCK_PROJECTS = {
    "all": [
        {"id": "P001", "name": "智能客服平台", "status": "进行中", "progress": 65, "team": "工程部"},
        {"id": "P002", "name": "数据中台建设", "status": "进行中", "progress": 30, "team": "工程部"},
        {"id": "P003", "name": "官网改版", "status": "规划中", "progress": 0, "team": "产品部"},
        {"id": "P004", "name": "Q3 营销活动", "status": "规划中", "progress": 10, "team": "市场部"},
        {"id": "P005", "name": "员工培训系统", "status": "已完成", "progress": 100, "team": "人事部"},
    ]
}


class GetProjects(BaseTool):
    """查询公司项目列表"""

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
                    "enum": ["进行中", "规划中", "已完成"],
                }
            },
            "required": [],
        }

    async def __call__(self, status: Optional[str] = None, **kwargs) -> str:
        params = {}
        if status:
            params["status"] = status

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base_url}/api/projects",
                    params=params,
                    timeout=5.0,
                )
                data = resp.json()
        except Exception:
            return self._mock_response(status)

        projects = data.get("projects", [])
        if not projects:
            return f"未找到状态为'{status or '所有'}的项目"

        lines = [f"共找到 {data.get('total', len(projects))} 个项目:"]
        for proj in projects:
            lines.append(
                f"  - [{proj['id']}] {proj['name']} | {proj['status']} "
                f"| 进度{proj['progress']}% | {proj['team']}"
            )
        return "\n".join(lines)

    def _mock_response(self, status: Optional[str] = None) -> str:
        """服务器不可用时返回模拟数据"""
        if status:
            data = [p for p in MOCK_PROJECTS["all"] if p["status"] == status]
        else:
            data = MOCK_PROJECTS["all"]
        if not data:
            return f"未找到状态为'{status or '所有'}的项目"
        lines = [f"共找到 {len(data)} 个项目（模拟数据）:"]
        for proj in data:
            lines.append(
                f"  - [{proj['id']}] {proj['name']} | {proj['status']} "
                f"| 进度{proj['progress']}% | {proj['team']}"
            )
        return "\n".join(lines)
