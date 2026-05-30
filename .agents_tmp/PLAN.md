# 1. OBJECTIVE

创建一个 Jupyter Notebook 教程 `tavily_agent_tutorial.ipynb`，逐步实现 RAGFlow 中 Tavily 工具调用的完整流程（思维链 + 多任务并行执行）。

Notebook 文件路径: `/workspace/project/notebook/tavily_agent_tutorial.ipynb`

# 2. CONTEXT SUMMARY

用户已确认核心代码路径：
- `rag/prompts/next_step.md` — 思维链提示词
- `rag/llm/chat_model.py:519-552` — 基础并行执行
- `rag/llm/chat_model.py:1784-1814` — LiteLLM 并行执行
- `agent/component/agent_with_tools.py:194-212` — Agent 推理集成
- `agent/tools/tavily.py:101-154` — 单个工具执行

# 3. APPROACH OVERVIEW

Notebook 包含 7 个部分：环境准备 → 工具基类 → Tavily 工具 → 思维链提示词 → LLM 并行执行 → Agent 推理循环 → 完整运行示例。每个部分包含 Markdown 说明（介绍架构和设计点）和可执行代码。

# 4. IMPLEMENTATION STEPS

## Step 0: 创建 notebook 骨架

创建一个 Jupyter Notebook 文件，包含以下 cell 结构：

### Cell 1 (markdown): 标题页
```markdown
# 📚 RAGFlow Tavily Agent 完整实现教程

本教程逐步教你实现类似 RAGFlow 的 Tavily 搜索工具 + 思维链推理 + 多任务并行执行 系统。

## 🎯 学习目标
- 理解 RAGFlow 中工具调用系统的架构设计
- 掌握 LLM Function Calling 的工具注册与执行机制
- 实现思维链（Chain of Thought）驱动的 Agent 推理
- 实现多个搜索任务的并行执行

## 🔗 对应 RAGFlow 源码路径

| 本教程步骤 | RAGFlow 对应代码 |
|-----------|-----------------|
| Step 1: 工具基类 | agent/tools/tavily.py (部分) |
| Step 2: Tavily 工具 | agent/tools/tavily.py:101-154 |
| Step 3: 思维链提示词 | rag/prompts/next_step.md (全文) |
| Step 4: 并行执行 | rag/llm/chat_model.py:519-552, 1784-1814 |
| Step 5: Agent 推理 | agent/component/agent_with_tools.py:194-212 |
| Step 6: 完整运行 | 综合以上所有模块 |
```

### Cell 2 (markdown): Step 0 标题
```markdown
---
## 📦 Step 0: 环境准备
首先安装必要的依赖库。
```

### Cell 3 (code): 安装依赖
```python
# 安装依赖（取消注释以安装）
# !pip install openai aiohttp nest_asyncio
```

### Cell 4 (code): 导入基础库
```python
import json
import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

print("✅ 基础依赖导入成功")
```

### Cell 5 (markdown): Step 1 标题
```markdown
---
## 🏗️ Step 1: 实现工具基类与注册机制

> **对应 RAGFlow**: `agent/tools/tavily.py` (基础部分)

### 架构说明

RAGFlow 使用工具注册表模式来管理所有可用工具。每个工具必须提供：
1. 名称 (name): 唯一标识符，LLM 通过此名称调用工具
2. 描述 (description): 告诉 LLM 工具的功能
3. 参数定义 (parameters): JSON Schema 格式，描述工具需要的参数
4. 执行逻辑 (__call__): 工具的实际执行代码

这种设计的精妙之处在于：工具定义与执行分离。定义部分告诉 LLM "你能做什么"，执行部分完成实际工作。
```

### Cell 6 (code): 实现 BaseTool 和 ToolRegistry
```python
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
        RAGFlow 中对应的转换逻辑在 agent/component/ 中。
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
        print(f"📝 已注册工具: {tool.name}")
    
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


# 创建全局工具注册表
registry = ToolRegistry()
print("✅ 工具基类和注册表已实现")
```

### Cell 7 (markdown): Step 1 设计点说明
```markdown
### 💡 关键设计点

- to_openai_tool() 方法: 将 Python 工具类转换为 OpenAI Function Calling 的 JSON 格式。这是 LLM 理解工具能力的桥梁。
- 注册表模式: 用字典存储工具，支持按名称快速查找。RAGFlow 实际使用了更复杂的注册机制（支持动态加载）。
- 抽象基类: 强制子类实现 name, description, parameters, __call__，保证所有工具接口一致。
```

### Cell 8 (markdown): Step 2 标题
```markdown
---
## 🔍 Step 2: 实现 Tavily 搜索工具

> **对应 RAGFlow**: `agent/tools/tavily.py:101-154`

### 架构说明

Tavily 是 RAGFlow 集成的搜索工具，用于获取实时网络信息。RAGFlow 的实现包括：
1. API 调用: 使用 aiohttp 异步调用 Tavily API
2. 参数处理: 支持 search_depth, max_results 等参数
3. 结果格式化: 将搜索结果格式化为文本返回给 LLM
```

### Cell 9 (code): 实现 TavilySearch 工具
```python
class TavilySearch(BaseTool):
    """
    Tavily 网络搜索工具。
    对应 RAGFlow: agent/tools/tavily.py (行 101-154)
    """
    
    TAVILY_API_URL = "https://api.tavily.com/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
    
    @property
    def name(self) -> str:
        return "tavily_search"
    
    @property
    def description(self) -> str:
        return (
            "Search the web for current information on any topic. "
            "Returns relevant search results with titles, URLs, and content snippets. "
            "Use this when you need up-to-date information from the internet."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and concise."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "Search depth: 'basic' for quick results, 'advanced' for comprehensive results",
                    "default": "basic"
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of domains to specifically include in results"
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of domains to specifically exclude from results"
                }
            },
            "required": ["query"]
        }
    
    async def __call__(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        if not self.api_key:
            return self._mock_search(query, max_results)
        
        try:
            import aiohttp
            payload = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "api_key": self.api_key
            }
            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.TAVILY_API_URL, json=payload) as response:
                    if response.status != 200:
                        return f"Tavily API error: {response.status}"
                    data = await response.json()
            
            return self._format_results(data.get("results", []))
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _format_results(self, results: List[Dict]) -> str:
        if not results:
            return "No relevant search results found."
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"[{i}] {result.get('title', 'No title')}\n"
                f"    URL: {result.get('url', '')}\n"
                f"    Content: {result.get('content', '')}"
            )
        return "\n\n".join(formatted)
    
    def _mock_search(self, query: str, max_results: int = 5) -> str:
        mock_results = {
            "artificial intelligence": [
                {"title": "AI Development Trends 2024", "url": "https://example.com/ai-trends", "content": "Major advances in large language models and multimodal AI..."},
                {"title": "Machine Learning Algorithm Comparison", "url": "https://example.com/ml-comparison", "content": "Comparison of mainstream ML algorithms on standard datasets..."},
            ],
            "climate change": [
                {"title": "Global Climate Report", "url": "https://example.com/climate-report", "content": "Latest reports show continued rise in global average temperatures..."},
            ]
        }
        results = mock_results.get(query.lower(), [
            {"title": f"Search results for '{query}'", "url": "https://example.com/result", "content": f"This is a simulated search result for {query}..."}
        ])
        return self._format_results(results[:max_results])


# 注册 Tavily 搜索工具
tavily_tool = TavilySearch()
registry.register(tavily_tool)
print(f"✅ Tavily search tool implemented and registered")
print(f"   Name: {tavily_tool.name}")
print(f"   Description: {tavily_tool.description[:60]}...")
```

### Cell 10 (markdown): Step 2 设计点说明
```markdown
### 💡 关键设计点

- JSON Schema 参数定义: parameters 属性定义了工具参数的 JSON Schema。LLM 的 function calling 机制会根据这个 schema 自动验证参数。
- 异步 __call__: 使用 async def 让工具支持异步执行，这是实现并行搜索的关键。
- 结果格式化: _format_results 将结构化数据转换为 LLM 友好的文本格式。
- 模拟模式: _mock_search 允许在没有真实 API key 的情况下演示功能。
```

### Cell 11 (markdown): Step 3 标题
```markdown
---
## 🧠 Step 3: 实现思维链提示词

> **对应 RAGFlow**: `rag/prompts/next_step.md` (全文)

### 架构说明

思维链（Chain of Thought）提示词是 Agent 智能决策的核心。它告诉 LLM：
1. 当前处于什么状态
2. 已经获得了什么信息
3. 下一步应该做什么

RAGFlow 的 next_step.md 提示词指导 LLM 在每个推理步骤中进行思考。
```

### Cell 12 (code): 实现思维链提示词
```python
# 系统提示词 — 定义 Agent 的行为框架
SYSTEM_PROMPT = """You are an AI research assistant with web search capabilities. 
Your goal is to provide comprehensive and accurate answers to user questions.

## Available Tools

You have access to the following tools:

- tavily_search: Search the web for current information. Use this when you need 
  up-to-date information or want to verify facts.

## How to Work

For each step, follow this process:

1. THINK: Analyze the current situation:
   - What has the user asked?
   - What information do I already have?
   - What information am I still missing?

2. DECIDE: Choose the next action:
   - If you need more information → Call tavily_search with specific queries
   - If you have enough information → Provide the final answer

3. PLAN: If using search, think about:
   - What specific queries would give the most relevant results?
   - Should I search for multiple aspects at once? (You can call search multiple times in parallel)

## Important Guidelines

- Always search for specific, focused queries rather than broad topics
- When you need information on multiple aspects, make parallel search calls
- Base your decisions on the information you've already gathered
- Be thorough but efficient in your research"""


# 下一步决策提示词 — 对应 RAGFlow 的 next_step.md
NEXT_STEP_PROMPT = """## Current Situation

User's question: {user_query}

{search_history}

## Your Turn

Think about:
1. What information do you already have?
2. What additional information do you need to answer the question well?
3. What search queries would help you find that information?

Then either:
- Call tavily_search tool(s) to gather more information
- Or provide your final answer if you have enough information"""


def build_next_step_prompt(
    user_query: str,
    search_history: Optional[str] = None
) -> str:
    """
    构建下一步的决策提示词。
    对应 RAGFlow: rag/prompts/next_step.md
    """
    if search_history:
        history_text = f"## Previous Search Results\n\n{search_history}"
    else:
        history_text = "You have not performed any searches yet."
    
    return NEXT_STEP_PROMPT.format(
        user_query=user_query,
        search_history=history_text
    )


print("✅ Chain of thought prompts implemented")
print(f"   System prompt length: {len(SYSTEM_PROMPT)} chars")
print(f"   Next step prompt template defined")
```

### Cell 13 (markdown): Step 3 设计点说明
```markdown
### 💡 关键设计点

- NEXT_STEP_PROMPT 模板: 使用 {user_query} 和 {search_history} 作为占位符，每次推理循环时动态填充。
- build_next_step_prompt(): 将状态信息编码为提示词，引导 LLM 思考。
- 搜索历史传递: 将之前的搜索结果传递给 LLM，让它基于已有信息做决策 — 这就是思维链的核心机制。
```

### Cell 14 (markdown): Step 4 标题
```markdown
---
## ⚡ Step 4: 实现 LLM 调用与并行执行

> **对应 RAGFlow**: 
> - rag/llm/chat_model.py:519-552（基础并行）
> - rag/llm/chat_model.py:1784-1814（LiteLLM 并行）

### 架构说明

这是整个系统中最精妙的部分。RAGFlow 实现了两层并行：

1. 基础并行 (chat_model.py:519-552): 处理 LLM 返回的多个 tool_calls，将它们并发执行
2. LiteLLM 并行 (chat_model.py:1784-1814): LiteLLM 后端的并行实现，兼容多种 LLM

并行执行的关键流程：
```
LLM 输出 → [tool_call_1, tool_call_2, tool_call_3]
                ↓            ↓            ↓
         task_1运行    task_2运行    task_3运行  ← 同时执行！
                ↓            ↓            ↓
         result_1    result_2    result_3
```

实现并行执行的核心 Python 原语是 asyncio.gather()。
```

### Cell 15 (code): 实现 LLMClient
```python
class LLMClient:
    """
    LLM 客户端封装 — 处理与 LLM 的对话。
    对应 RAGFlow: rag/llm/chat_model.py
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self._use_mock = not self.api_key
    
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
            client = AsyncOpenAI(api_key=self.api_key)
            
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
            print(f"LLM API error: {e}")
            self._use_mock = True
            return self._mock_chat(messages)
    
    def _mock_chat(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """模拟 LLM 响应（用于演示）"""
        last_msg = messages[-1].get("content", "")
        has_results = "Search Results" in last_msg or "previous" in last_msg.lower()
        
        if has_results:
            return (
                "Based on the search results, here is my comprehensive answer:\n\n"
                "This is a simulated response. In actual usage, the LLM would analyze "
                "the real search results and generate a more comprehensive answer."
            ), []
        else:
            return "", [
                {
                    "id": "call_mock_1",
                    "name": "tavily_search",
                    "arguments": {"query": "artificial intelligence", "max_results": 2}
                },
                {
                    "id": "call_mock_2",
                    "name": "tavily_search",
                    "arguments": {"query": "machine learning", "max_results": 2}
                }
            ]


print("✅ LLM client implemented")
```

### Cell 16 (code): 实现并行执行器
```python
async def execute_tools_parallel(
    tool_calls: List[Dict],
    registry: ToolRegistry
) -> List[Dict]:
    """
    并行执行多个工具调用。
    对应 RAGFlow: rag/llm/chat_model.py:519-552（基础并行）
    
    这是多任务并行执行的核心！
    """
    if not tool_calls:
        return []
    
    async def execute_one(tc: Dict) -> Dict:
        """执行单个工具调用"""
        tool_name = tc["name"]
        args = tc.get("arguments", {})
        
        tool = registry.get(tool_name)
        if not tool:
            return {"tool_call_id": tc.get("id", ""), "name": tool_name, "result": f"Unknown tool: {tool_name}"}
        
        print(f"   Executing: {tool_name}({json.dumps(args, ensure_ascii=False)})")
        result = await tool(**args)
        
        return {"tool_call_id": tc.get("id", ""), "name": tool_name, "result": result}
    
    # ===== 关键：使用 asyncio.gather 并行执行 =====
    print(f"   Parallel execution of {len(tool_calls)} tool calls...")
    start_time = time.time()
    
    tasks = [execute_one(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = time.time() - start_time
    print(f"   All tools completed in {elapsed:.2f}s")
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "tool_call_id": tool_calls[i].get("id", ""),
                "name": tool_calls[i]["name"],
                "result": f"Execution error: {str(result)}"
            })
        else:
            processed_results.append(result)
    
    return processed_results


print("✅ Parallel executor implemented")
print("   Core: execute_tools_parallel() uses asyncio.gather")
```

### Cell 17 (markdown): Step 4 设计点说明
```markdown
### 💡 关键设计点

- asyncio.gather(*tasks): 这是 Python 并发执行多个异步任务的标准方式。所有任务同时开始，等待最慢的那个完成。
- 并行 vs 串行: 如果串行执行 3 个各需 2 秒的搜索，需要 6 秒；并行执行只需约 2 秒。
- return_exceptions=True: 防止一个任务失败导致所有任务取消。
- RAGFlow 的实现: 在 chat_model.py:519-552 中使用了类似的 asyncio.gather 模式，但在 1784-1814 行还处理了 LiteLLM 的兼容性。
```

### Cell 18 (markdown): Step 5 标题
```markdown
---
## 🔄 Step 5: 实现 Agent 推理循环

> **对应 RAGFlow**: `agent/component/agent_with_tools.py:194-212`

### 架构说明

Agent 推理循环是整个系统的大脑。它实现了 ReAct 模式（Reasoning + Acting）：
```
思考 → 行动 → 观察 → 思考 → 行动 → 观察 → ... → 最终答案
```

RAGFlow 的 agent_with_tools.py:194-212 正是实现了这个循环。
```

### Cell 19 (code): 实现 AgentWithTools
```python
class AgentWithTools:
    """
    Agent 推理循环 — 整合 LLM 推理与工具调用。
    对应 RAGFlow: agent/component/agent_with_tools.py:194-212
    """
    
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 5
    ):
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
    
    async def run(self, user_query: str) -> str:
        """
        执行 Agent 推理循环。
        对应 RAGFlow: agent/component/agent_with_tools.py:194-212 的核心逻辑
        """
        print(f"\n{'='*60}")
        print(f"Agent starting")
        print(f"   Query: {user_query}")
        print(f"{'='*60}\n")
        
        messages = [{"role": "system", "content": self.system_prompt}]
        search_history = ""
        
        for step in range(1, self.max_steps + 1):
            print(f"\n{'─'*40}")
            print(f"Step {step}/{self.max_steps}")
            print(f"{'─'*40}")
            
            # 1. 构建当前步提示词（思维链）
            user_prompt = build_next_step_prompt(user_query, search_history)
            messages.append({"role": "user", "content": user_prompt})
            
            # 2. 调用 LLM 获取决策
            print(f"   Calling LLM...")
            response_text, tool_calls = await self.llm.chat(
                messages=messages,
                tools=self.registry.to_openai_tools()
            )
            
            # 3. 检查 LLM 是否返回了最终答案
            if not tool_calls:
                print(f"   LLM returned final answer")
                print(f"{'='*60}")
                print(f"Final Answer:")
                print(f"{response_text}")
                print(f"{'='*60}")
                return response_text or "No answer generated"
            
            # 4. LLM 决定调用工具
            print(f"   LLM calling {len(tool_calls)} tool(s):")
            for tc in tool_calls:
                print(f"      - {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
            
            # 5. 并行执行所有工具调用
            results = await execute_tools_parallel(tool_calls, self.registry)
            
            # 6. 将工具结果添加到对话历史
            for result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["result"]
                })
                search_history += f"\n### Search: {result['name']}\n"
                search_history += f"Result: {result['result'][:200]}...\n"
        
        print(f"\nReached max steps ({self.max_steps})")
        return "Reached maximum reasoning steps."


print("✅ Agent reasoning loop implemented")
print("   Core: AgentWithTools.run() - ReAct reasoning loop")
```

### Cell 20 (markdown): Step 5 设计点说明
```markdown
### 💡 关键设计点

- 循环结构: for step in range(1, max_steps + 1) 实现了 ReAct 循环。
- 思维链传递: 每次循环调用 build_next_step_prompt()，将搜索历史传入，让 LLM "记住"之前的搜索结果。
- 工具结果回传: 将工具执行结果以 role: "tool" 格式添加回 messages，LLM 能看到这些结果。
- 终止条件: 当 LLM 不再返回 tool_calls 时，说明它认为信息足够，返回最终答案。
- 安全机制: max_steps 防止 LLM 陷入无限搜索循环。
```

### Cell 21 (markdown): Step 6 标题
```markdown
---
## 🎬 Step 6: 完整运行示例

现在让我们把所有组件串联起来，运行一个完整的示例！

### 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentWithTools                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  思维链提示词  │───▶│  LLM Client  │───▶│ 并行执行器    │  │
│  │ next_step.md │    │ chat()       │    │ execute_parallel│  │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘  │
│                             │                   │          │
│                      tool_calls           执行结果           │
│                             │                   │          │
│                             ▼                   ▼          │
│                    ┌────────────────────────────────┐      │
│                    │        ToolRegistry            │      │
│                    │  ┌──────────┐ ┌──────────┐    │      │
│                    │  │ Tavily   │ │ 其他工具  │    │      │
│                    │  │ Search   │ │ ...      │    │      │
│                    │  └──────────┘ └──────────┘    │      │
│                    └────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```
```

### Cell 22 (code): 运行示例
```python
async def main():
    """完整的 Agent 运行示例"""
    print("Initializing components...")
    llm = LLMClient()
    agent = AgentWithTools(
        llm=llm,
        registry=registry,
        max_steps=3
    )
    
    user_query = "What are the latest developments in AI and machine learning in 2024?"
    result = await agent.run(user_query)
    return result

# Run in Jupyter
import nest_asyncio
try:
    nest_asyncio.apply()
except:
    pass

result = await main()
print(f"\nExecution complete!")
```

### Cell 23 (markdown): 总结页
```markdown
---
## 📊 总结：代码组织与 RAGFlow 对照

### 文件映射关系

| 本教程模块 | 对应 RAGFlow 文件 | 行号 | 核心功能 |
|-----------|------------------|------|---------|
| BaseTool, ToolRegistry | agent/tools/tavily.py | 1-100 | 工具基类和注册 |
| TavilySearch | agent/tools/tavily.py | 101-154 | 搜索工具执行 |
| SYSTEM_PROMPT, NEXT_STEP_PROMPT | rag/prompts/next_step.md | 全文 | 思维链提示词 |
| execute_tools_parallel() | rag/llm/chat_model.py | 519-552 | 基础并行执行 |
| LLMClient.chat() | rag/llm/chat_model.py | 1784-1814 | LiteLLM 并行 |
| AgentWithTools.run() | agent/component/agent_with_tools.py | 194-212 | Agent 推理集成 |

### 三个核心问题的答案

1. 大模型实现 Tavily 工具调用的代码
   - 工具定义（name, description, parameters）+ to_openai_tool() 转换
   - LLM 通过 function calling 机制决定调用哪个工具
   - ToolRegistry 查找并执行对应工具

2. 思维链 + 多任务并行执行的代码
   - 思维链：next_step.md 提示词 + 搜索历史传递
   - 并行执行：execute_tools_parallel() 使用 asyncio.gather()
   - 关键：LLM 一次返回多个 tool_calls，执行器并发运行

3. LLM 思维链如何做工具调用决策
   - 系统提示词定义工具使用说明
   - 每步提示词包含当前状态（已有信息 + 用户问题）
   - LLM 基于这些信息自主决定：是否需要搜索 → 搜索什么 → 是否继续
   - 多轮迭代中，决策基于逐步积累的信息（这就是思维链的力量）

### 关键设计模式

1. 注册表模式: 工具通过注册表管理，支持动态扩展
2. 工厂模式: to_openai_tool() 将工具转换为 LLM 可理解的格式
3. 策略模式: 思维链提示词定义了推理策略，LLM 执行具体推理
4. 异步并行: 使用 asyncio.gather 实现真正的并行搜索
5. ReAct 循环: 思考→行动→观察的迭代推理模式
```

## 创建方法

使用 Python 脚本创建 notebook：

```python
import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Helper to add cells
def add_cell(cell_type, source):
    nb["cells"].append({
        "cell_type": cell_type,
        "execution_count": None if cell_type == "code" else None,
        "metadata": {},
        "outputs": [] if cell_type == "code" else [],
        "source": source.split('\n') if isinstance(source, str) else source
    })

# Add all cells as described in steps above
# ... (implement all 23 cells)

with open('/workspace/project/notebook/tavily_agent_tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
```

# 5. TESTING AND VALIDATION

通过运行 notebook 中的代码示例验证每个步骤的正确性：
1. 逐 cell 执行 notebook，确保没有语法错误
2. 执行 Step 6 的完整示例，验证 Agent 能正常运行（使用 mock 模式）
3. 如果设置了 OPENAI_API_KEY 和 TAVILY_API_KEY，验证真实 API 调用
