
"""Agent 推理循环 — ReAct 模式实现"""
import json
import time
import asyncio
from typing import List, Dict

from ..llm import LLMClient
from ..tools import ToolRegistry
from ..prompts import SYSTEM_PROMPT, build_next_step_prompt


class AgentWithTools:
    """Agent 推理循环"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 5,
    ):
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    async def run(self, user_query: str) -> str:
        """
        执行 Agent 推理循环。

        流程：
        1. 构建提示词，调用 LLM
        2. LLM 返回 tool_calls → 并行执行工具 → 结果加入对话历史
        3. 重复直到 LLM 给出最终答案或达到最大步骤
        """
        print(f"\n{'='*60}")
        print(f"🤖 Agent 启动")
        print(f"   问题: {user_query}")
        print(f"{'='*60}\n")

        # 初始化对话历史
        messages = [{"role": "system", "content": self.system_prompt}]
        search_history = ""

        for step in range(1, self.max_steps + 1):
            print(f"\n{'─'*50}")
            print(f"📍 第 {step}/{self.max_steps} 步")
            print(f"{'─'*50}")

            # 1. 构建当前轮的提示词
            user_prompt = build_next_step_prompt(user_query, search_history)
            messages.append({"role": "user", "content": user_prompt})

            # 2. 调用 LLM
            print(f"   🧠 调用 LLM 决策...")
            response_text, tool_calls = await self.llm.chat(
                messages=messages,
                tools=self.registry.to_openai_tools(),
            )

            # 3. 判断：LLM 是否调用了工具？
            if not tool_calls:
                # 没有 tool_calls → LLM 给出了最终答案
                print(f"   ✅ LLM 返回了最终答案")
                print(f"{'='*60}")
                print(f"💬 {response_text or '未生成回答'}")
                print(f"{'='*60}")
                return response_text or "未生成回答"

            # 4. LLM 决定调用工具 → 打印决策
            print(f"   🔧 LLM 决定调用 {len(tool_calls)} 个工具:")
            for tc in tool_calls:
                args = json.dumps(tc["arguments"], ensure_ascii=False)
                print(f"      - {tc['name']}({args})")

            # 5. 并行执行所有工具
            results = await self._execute_tools_parallel(tool_calls)

            # 6. 将工具结果加入对话历史（LLM 下一轮会看到这些结果）
            for result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["result"],
                })
                search_history += f"\n### {result['name']}\n"
                search_history += f"结果: {result['result'][:200]}\n"

        print(f"\n⚠️ 达到最大步骤数 ({self.max_steps})，未能生成回答")
        return "已达到最大推理步骤。"

    async def _execute_tools_parallel(self, tool_calls: List[Dict]) -> List[Dict]:
        """并行执行多个工具调用"""
        if not tool_calls:
            return []

        async def execute_one(tc: Dict) -> Dict:
            tool_name = tc["name"]
            args = tc.get("arguments", {})

            tool = self.registry.get(tool_name)
            if not tool:
                return {
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "result": f"未知工具: {tool_name}",
                }

            print(f"   🚀 执行: {tool_name}({json.dumps(args, ensure_ascii=False)})")
            result = await tool(**args)
            return {
                "tool_call_id": tc.get("id", ""),
                "name": tool_name,
                "result": result,
            }

        print(f"   ⚡ 并行执行 {len(tool_calls)} 个工具调用...")
        start_time = time.time()

        # 关键：asyncio.gather 让多个工具同时运行
        tasks = [execute_one(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time
        print(f"   ✅ 全部完成，耗时 {elapsed:.2f}s")

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "tool_call_id": tool_calls[i].get("id", ""),
                    "name": tool_calls[i]["name"],
                    "result": f"执行错误: {str(result)}",
                })
            else:
                processed_results.append(result)

        return processed_results
