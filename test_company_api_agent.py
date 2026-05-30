"""
测试 Notebook 核心逻辑 — 验证 Agent 能正常运行
"""

import json
import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

API_BASE_URL = "http://localhost:8080"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# ===================== 工具基类 =====================

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass
    @property
    @abstractmethod
    def description(self) -> str: pass
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]: pass
    @abstractmethod
    async def __call__(self, **kwargs) -> str: pass
    def to_openai_tool(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        print(f"📝 已注册工具: {tool.name}")
    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)
    def get_all(self) -> List[BaseTool]:
        return list(self._tools.values())
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

registry = ToolRegistry()

# ===================== 工具实现 =====================

import httpx

class GetEmployees(BaseTool):
    @property
    def name(self) -> str: return "get_employees"
    @property
    def description(self) -> str: return "查询公司员工信息列表。可按部门筛选员工。"
    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"department": {"type": "string", "description": "可选，按部门筛选", "enum": ["工程部", "产品部", "市场部", "人事部"]}}, "required": []}
    async def __call__(self, department: Optional[str] = None, **kwargs) -> str:
        params = {"department": department} if department else {}
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_BASE_URL}/api/employees", params=params)
            data = resp.json()
        lines = [f"共找到 {data['total']} 名员工:"]
        for emp in data["employees"]:
            lines.append(f"  - {emp['name']} | {emp['department']} | {emp['position']}")
        return "\n".join(lines)

class GetProjects(BaseTool):
    @property
    def name(self) -> str: return "get_projects"
    @property
    def description(self) -> str: return "查询公司项目列表。可按状态筛选。"
    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"status": {"type": "string", "description": "可选，按状态筛选", "enum": ["进行中", "规划中", "已完成"]}}, "required": []}
    async def __call__(self, status: Optional[str] = None, **kwargs) -> str:
        params = {"status": status} if status else {}
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_BASE_URL}/api/projects", params=params)
            data = resp.json()
        lines = [f"共找到 {data['total']} 个项目:"]
        for proj in data["projects"]:
            lines.append(f"  - [{proj['id']}] {proj['name']} | {proj['status']} | 进度{proj['progress']}% | {proj['team']}")
        return "\n".join(lines)

class GetMetrics(BaseTool):
    @property
    def name(self) -> str: return "get_metrics"
    @property
    def description(self) -> str: return "获取公司关键业务指标，包括营收、活跃用户数、用户满意度、系统可用性等。"
    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}
    async def __call__(self, **kwargs) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_BASE_URL}/api/metrics")
            data = resp.json()
        rev = data["revenue"]
        return (f"Q3 营收: {rev['q3']:,} {rev['unit']} | Q4 目标: {rev['q4_target']:,} {rev['unit']}\n"
                f"活跃用户: {data['users']['active']:,} | 本月新增: {data['users']['new_this_month']:,}\n"
                f"用户满意度: {data['satisfaction']}/5.0 | 系统可用率: {data['uptime']}%")

class SearchCompany(BaseTool):
    @property
    def name(self) -> str: return "search_company"
    @property
    def description(self) -> str: return "在公司内部知识库中搜索信息。"
    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}
    async def __call__(self, query: str, **kwargs) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_BASE_URL}/api/search?query={query}")
            data = resp.json()
        lines = [f"搜索 '{query}' 找到 {len(data['results'])} 条结果:"]
        for i, r in enumerate(data["results"], 1):
            lines.append(f"  {i}. {r}")
        return "\n".join(lines)

# 注册所有工具
for tool_cls in [GetEmployees, GetProjects, GetMetrics, SearchCompany]:
    registry.register(tool_cls())

# ===================== 提示词 =====================

SYSTEM_PROMPT = """你是一个公司内部智能助手，可以调用公司内部 API 来回答员工的问题。

可用工具:
- get_employees: 查询员工信息
- get_projects: 查询项目信息  
- get_metrics: 获取公司关键业务指标
- search_company: 内部搜索

工作流程: 分析问题 → 选择工具 → 综合信息生成回答"""

NEXT_STEP_PROMPT = """用户问题: {user_query}

{search_history}

请调用合适的工具来获取信息，或者如果信息已足够，给出最终回答。"""

def build_next_step_prompt(user_query: str, search_history: Optional[str] = None) -> str:
    if search_history:
        history_text = f"## 已获取的信息\n\n{search_history}"
    else:
        history_text = "你还没有获取任何信息。"
    return NEXT_STEP_PROMPT.format(user_query=user_query, search_history=history_text)

# ===================== LLM 客户端 =====================

class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._use_mock = not self.api_key
        mode = "模拟模式" if self._use_mock else "真实 API 模式"
        print(f"🤖 LLM 客户端: {DEEPSEEK_MODEL} — {mode}")
    
    async def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict]]:
        if self._use_mock:
            return self._mock_chat(messages)
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)
            kwargs = {"model": DEEPSEEK_MODEL, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)})
            return message.content or "", tool_calls
        except Exception as e:
            print(f"⚠️ LLM API 错误: {e}")
            self._use_mock = True
            return self._mock_chat(messages)
    
    def _mock_chat(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        last_msg = messages[-1].get("content", "")
        has_results = "已获取" in last_msg or "结果" in last_msg
        if has_results:
            return ("根据查询到的信息，我来为您总结：\n\n"
                    "（这是模拟回答，实际使用时 DeepSeek-V4-Flash 会基于真实 API 结果生成回答。）"), []
        else:
            return "", [
                {"id": "call_mock_1", "name": "get_employees", "arguments": {}},
                {"id": "call_mock_2", "name": "get_projects", "arguments": {"status": "进行中"}}
            ]

# ===================== 并行执行器 =====================

async def execute_tools_parallel(tool_calls: List[Dict], registry: ToolRegistry) -> List[Dict]:
    if not tool_calls:
        return []
    async def execute_one(tc: Dict) -> Dict:
        tool_name = tc["name"]
        args = tc.get("arguments", {})
        tool = registry.get(tool_name)
        if not tool:
            return {"tool_call_id": tc.get("id", ""), "name": tool_name, "result": f"未知工具: {tool_name}"}
        print(f"   🚀 执行: {tool_name}({json.dumps(args, ensure_ascii=False)})")
        result = await tool(**args)
        return {"tool_call_id": tc.get("id", ""), "name": tool_name, "result": result}
    
    print(f"   ⚡ 并行执行 {len(tool_calls)} 个工具调用...")
    start_time = time.time()
    tasks = [execute_one(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start_time
    print(f"   ✅ 全部完成，耗时 {elapsed:.2f}s")
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({"tool_call_id": tool_calls[i].get("id", ""), "name": tool_calls[i]["name"], "result": f"执行错误: {str(result)}"})
        else:
            processed_results.append(result)
    return processed_results

# ===================== Agent 推理循环 =====================

class AgentWithTools:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, system_prompt: str = SYSTEM_PROMPT, max_steps: int = 5):
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
    
    async def run(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"Agent 启动")
        print(f"   问题: {user_query}")
        print(f"{'='*60}\n")
        
        messages = [{"role": "system", "content": self.system_prompt}]
        search_history = ""
        
        for step in range(1, self.max_steps + 1):
            print(f"\n{'─'*50}")
            print(f"第 {step}/{self.max_steps} 步")
            print(f"{'─'*50}")
            
            user_prompt = build_next_step_prompt(user_query, search_history)
            messages.append({"role": "user", "content": user_prompt})
            
            print(f"   🤖 调用 LLM...")
            response_text, tool_calls = await self.llm.chat(messages=messages, tools=self.registry.to_openai_tools())
            
            if not tool_calls:
                print(f"   ✅ LLM 返回了最终答案")
                print(f"{'='*60}")
                print(f"最终回答:\n{response_text}")
                print(f"{'='*60}")
                return response_text or "未生成回答"
            
            print(f"   📋 LLM 决定调用 {len(tool_calls)} 个工具:")
            for tc in tool_calls:
                print(f"      - {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
            
            results = await execute_tools_parallel(tool_calls, self.registry)
            
            for result in results:
                messages.append({"role": "tool", "tool_call_id": result["tool_call_id"], "content": result["result"]})
                search_history += f"\n### {result['name']}\n"
                search_history += f"结果: {result['result'][:200]}...\n"
        
        print(f"\n⚠️ 达到最大步骤数 ({self.max_steps})")
        return "已达到最大推理步骤。"

# ===================== 运行测试 =====================

async def main():
    print("正在初始化组件...\n")
    llm = LLMClient()
    agent = AgentWithTools(llm=llm, registry=registry, max_steps=3)
    
    query = "我们公司有哪些进行中的项目？工程部有哪些员工？"
    result = await agent.run(query)
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\n✅ 测试完成！")
