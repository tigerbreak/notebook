
"""LLM 客户端 — 使用 OpenAI 兼容协议
支持真实 API 和模拟模式（无 API Key 时自动降级）"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple


class LLMClient:
    """LLM 客户端"""

    BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    MODEL = "deepseek-v4-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # 优先级：构造函数参数 > 环境变量 > 空字符串（模拟模式）
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model or self.MODEL
        self.base_url = base_url or self.BASE_URL
        self._use_mock = not self.api_key

    @property
    def mode(self) -> str:
        return "🧪 模拟模式" if self._use_mock else "🔌 真实 API 模式"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        与 LLM 对话。

        Args:
            messages: 对话历史 [{role, content}, ...]
            tools: 工具定义列表（OpenAI 格式）

        Returns:
            (回复文本, tool_calls 列表)
            - 如果 LLM 决定调用工具，text 通常为空，tool_calls 有内容
            - 如果 LLM 直接回答，text 有内容，tool_calls 为空
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
                        "arguments": json.loads(tc.function.arguments),
                    })

            return message.content or "", tool_calls

        except Exception as e:
            print(f"⚠️ LLM API 错误: {e}，切换到模拟模式")
            self._use_mock = True
            return self._mock_chat(messages)

    def _mock_chat(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """模拟 LLM 响应（用于无 API Key 时）"""
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
                {"id": "call_mock_1", "name": "get_employees", "arguments": {}},
                {
                    "id": "call_mock_2",
                    "name": "get_projects",
                    "arguments": {"status": "进行中"},
                },
            ]
