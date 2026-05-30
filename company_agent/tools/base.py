"""
工具基类与注册表
对应 RAGFlow: agent/tools/ 中的基础结构
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTool(ABC):
    """
    工具基类 — 所有工具必须继承此类。
    
    RAGFlow 中的设计：
    - name: 工具的唯一标识
    - description: 工具的功能描述（会传给 LLM）
    - parameters: 工具的参数定义（JSON Schema，会传给 LLM）
    - __call__: 工具的执行逻辑
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具的唯一标识符"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具的功能描述（提供给 LLM 参考）"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """工具的参数定义（JSON Schema 格式）"""
        pass
    
    @abstractmethod
    async def __call__(self, **kwargs) -> str:
        """执行工具的具体逻辑"""
        pass
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        将工具转换为 OpenAI Function Calling 格式。
        这是关键方法！它把 Python 类变成 LLM 能理解的格式。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """
    工具注册表 — 管理所有可用工具。
    RAGFlow 中对应 agent/component/__init__.py 的注册逻辑。
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """注册一个工具"""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        return self._tools.get(name)
    
    def get_all(self) -> List[BaseTool]:
        """获取所有已注册的工具"""
        return list(self._tools.values())
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """将所有工具转换为 OpenAI 格式"""
        return [tool.to_openai_tool() for tool in self._tools.values()]
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
