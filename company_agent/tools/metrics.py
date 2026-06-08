
"""公司指标工具 — 调用 /api/metrics"""
from typing import Any, Dict
import httpx
from .base import BaseTool


# 模拟指标数据（服务器不可用时使用）
MOCK_METRICS = {
    "metrics": {
        "Q3 营收": "¥1,280 万",
        "员工总数": "187 人",
        "满意度评分": "4.6/5.0",
        "进行中项目": "5 个",
        "本月入职": "12 人",
    }
}


class GetMetrics(BaseTool):
    """获取公司关键业务指标"""

    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "get_metrics"

    @property
    def description(self) -> str:
        return "获取公司关键业务指标，包括营收、员工数、满意度等"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def __call__(self, **kwargs) -> str:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base_url}/api/metrics",
                    timeout=5.0,
                )
                data = resp.json()
        except Exception:
            return self._mock_response()

        lines = ["📊 公司关键指标:"]
        for key, val in data.get("metrics", {}).items():
            lines.append(f"  - {key}: {val}")
        return "\n".join(lines)

    def _mock_response(self) -> str:
        """服务器不可用时返回模拟数据"""
        lines = ["📊 公司关键指标（模拟数据）:"]
        for key, val in MOCK_METRICS["metrics"].items():
            lines.append(f"  - {key}: {val}")
        return "\n".join(lines)
