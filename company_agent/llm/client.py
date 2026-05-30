"""
LLM 客户端模块
对应 RAGFlow: rag/llm/chat_model.py
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class LLMClient:
    """
    LLM 客户端 — 使用 DeepSeek-V4-Flash
    
    对应 RAGFlow: rag/llm/chat_model.py 中的 chat 方法
    """
    
    BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    MODEL = "deepseek-v4-flash"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        # 优先级：构造函数参数 > 环境变量 > 空字符串（模拟模式）
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model or self.MODEL
        self.base_url = base_url or self.BASE_URL
        self._use_mock = not self.api_key
    
    @property
    def mode(self) -> str:
        return "模拟模式" if self._use_mock else "真实 API 模式"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, List[Dict]]:
        """
        与 LLM 对话并获取响应。
        对应 RAGFlow: rag/llm/chat_model.py 中的 chat 方法
        """
        if self._use_mock:
            return self._mock_chat(messages)
        
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            
            kwargs = {"model": self.model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            
            response = await client.chat.completions.create(**kwargs)
            
            choice = response.choices[0]
            message = choice.message
            
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    })
            
            return message.content or "", tool_calls
        except Exception as e:
            print(f"⚠️ LLM API 错误: {e}")
            self._use_mock = True
            return self._mock_chat(messages)
    
    def _mock_chat(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """模拟 LLM 响应（用于无 API Key 时的演示）"""
        last_msg = messages[-1].get("content", "")
        has_results = "已获取" in last_msg or "结果" in last_msg
        
        if has_results:
            return (
                "根据查询到的信息，我来为您总结：\n\n"
                "这是模拟 LLM 生成的回答。在实际使用时，DeepSeek-V4-Flash "
                "会分析真实的 API 结果并生成更全面的回答。"
            ), []
        else:
            return "", [
                {
                    "id": "call_mock_1",
                    "name": "get_employees",
                    "arguments": {}
                },
                {
                    "id": "call_mock_2",
                    "name": "get_projects",
                    "arguments": {"status": "进行中"}
                }
            ]
