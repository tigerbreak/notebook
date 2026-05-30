"""
主运行入口
"""

import os
import asyncio
from typing import Optional

from .tools import ToolRegistry
from .tools.employees import GetEmployees
from .tools.projects import GetProjects
from .tools.metrics import GetMetrics
from .tools.search import SearchCompany
from .llm import LLMClient
from .agent import AgentWithTools


def create_registry(api_base_url: str = "http://localhost:8080") -> ToolRegistry:
    """创建并注册所有工具的注册表"""
    registry = ToolRegistry()
    registry.register(GetEmployees(api_base_url=api_base_url))
    registry.register(GetProjects(api_base_url=api_base_url))
    registry.register(GetMetrics(api_base_url=api_base_url))
    registry.register(SearchCompany(api_base_url=api_base_url))
    return registry


async def run(
    query: str,
    api_key: Optional[str] = None,
    api_base_url: str = "http://localhost:8080",
    max_steps: int = 3
) -> str:
    """
    运行 Agent 的便捷函数
    
    Args:
        query: 用户问题
        api_key: LLM API Key（可选，不传则读环境变量 DEEPSEEK_API_KEY）
        api_base_url: 内部 API 地址
        max_steps: 最大推理步骤
    
    Returns:
        Agent 的最终回答
    """
    registry = create_registry(api_base_url=api_base_url)
    llm = LLMClient(api_key=api_key)
    agent = AgentWithTools(llm=llm, registry=registry, max_steps=max_steps)
    return await agent.run(query)


if __name__ == "__main__":
    asyncio.run(run("我们公司有哪些进行中的项目？工程部有哪些员工？"))