# 1. OBJECTIVE

在 `ai_tool.ipynb` 中追加企业级 RAG Tool 完整教程，参照 RAGFlow `tavily.py` 架构，覆盖从后端工具实现到前端交互的完整链路。

# 2. CONTEXT SUMMARY

- **已有**: `company_agent/` 已有 `BaseTool`+`ToolRegistry` 架构、4个API工具、Agent循环、Mock Server
- **参照源**: RAGFlow `agent/tools/tavily.py` 的 `ToolParamBase`+`ToolBase` 分离设计
- **目标文件**: `ai_tool.ipynb` 末尾追加 Step 7~10

# 3. APPROACH OVERVIEW

按 RAGFlow Tavily 模式新增 RAG 工具，包含 Param/Tool 分离、Mock RAG Server、Agent 注册集成、Notebook 教程 + 前端交互展示。

# 4. IMPLEMENTATION STEPS

## Step 1: 创建 `company_agent/tools/rag.py`
- **目标**: 参照 `TavilySearchParam`+`TavilySearch` 模式实现 `RAGSearch` 工具
- **方法**: 继承 `BaseTool`，参数: `query`(必需), `top_k`, `min_score`, `include_citations`；包含重试和错误处理

## Step 2: 扩展 `mock_api_server.py`
- **目标**: 新增 `POST /api/rag/search` 端点模拟向量检索
- **方法**: 内置文档 chunks，返回 content/source/page/score（参照 RAGFlow chunk 格式）

## Step 3: 注册 RAG 工具到 Agent
- **目标**: `main.py` 中注册 `RAGSearch`，更新 system prompt

## Step 4: 在 `ai_tool.ipynb` 追加教程 (Step 7~10)
- **目标**: 完整的 step-by-step 教程
- **内容**:
  - Step 7: RAG 工具原理 — RAGFlow 架构对比
  - Step 8: 实现代码展示
  - Step 9: 启动 Mock Server + 测试工具
  - Step 10: 端到端 Agent 运行，展示效果

## Step 5: 前端交互展示
- **目标**: notebook 内嵌 HTML/JS 问答界面
- **内容**: 用户提问 → LLM 调用 RAG → 检索结果 + 引用来源高亮

# 5. TESTING AND VALIDATION

1. 执行新增 cells，无语法错误
2. Mock RAG Server 返回正确格式
3. Agent 端到端运行验证（LLM 调用 RAG → 返回带引用回答）
4. 前端界面正常交互
