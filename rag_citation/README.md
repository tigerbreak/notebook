# RAGFlow 引用处理端到端流程复现

基于 RAGFlow 开源项目源码分析的 Jupyter Notebook 交互式练习，完整复现引用处理流程：

**前台提示词引导 → 模型生成 → 后端向量级校验对齐 → 结构化输出**

## 目录结构

```
rag_citation/
├── rag_citation_exercise.ipynb  # 主 Notebook 文件（21 cells）
├── test_chunks.json             # 测试数据（2 个知识块）
└── README.md                    # 本文件
```

## 环境配置

### 依赖安装

```bash
pip install openai numpy jinja2 jupyter
```

### 运行模式

Notebook 支持两种运行模式，通过 `USE_MOCK` 开关切换：

| 模式 | `USE_MOCK` | 依赖 | 说明 |
|------|-----------|------|------|
| **Mock 模式** | `True` | 仅 numpy | 本地模拟，无需外部 API |
| **真实 API 模式** | `False` | OpenAI 兼容 API | 需要 LLM 和 Embedding 服务 |

### 真实 API 配置

```bash
# LLM 配置（Ollama / vLLM / OpenAI 等）
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5:7b"

# Embedding 配置
export EMBEDDING_BASE_URL="http://localhost:11434/v1"
export EMBEDDING_API_KEY="ollama"
export EMBEDDING_MODEL="nomic-embed-text"
```

## 运行步骤

### 方式一：Jupyter Notebook

```bash
cd rag_citation
jupyter notebook rag_citation_exercise.ipynb
```

然后按顺序执行所有 Cell（从上到下），推荐使用 "Run All" 功能。

### 方式二：命令行执行

```bash
cd rag_citation
jupyter nbconvert --to notebook --execute --inplace rag_citation_exercise.ipynb
```

### 方式三：导出为 HTML

```bash
jupyter nbconvert --to html rag_citation_exercise.ipynb
```

## Notebook 结构

| Cell | 阶段 | 内容 |
|------|------|------|
| 1-3 | 环境初始化 | 安装依赖、导入库、定义客户端类 |
| 4-5 | 阶段一 | 加载测试数据、构建知识库上下文 (kb_prompt) |
| 6-7 | 阶段一 | 引用格式模板 (citation_prompt)、组装完整 Prompt |
| 8-9 | 阶段二 | LLM 生成原始回答、分析引用问题 |
| 10-14 | 阶段三 | `insert_citations()` 核心逻辑复现：代码块保护 → 断句 → 过滤 → 相似度 → 引用注入 |
| 15-16 | 阶段四 | 运行引用校验、构建标准 JSON 响应体 |
| 17-19 | 端到端测试 | Pipeline 函数、运行测试、对比分析 |
| 20 | 多语言测试 | 中文断句和引用对齐验证 |

## 核心函数说明

### `kb_prompt()`
将知识块格式化为 RAGFlow 风格的上下文，每个 chunk 以 `ID: X` 开头，包含 Title、Page、Content 字段。

### `citation_prompt()`
返回引用格式约束模板，规定 `[ID:x]` 的格式和使用规则。

### `build_rag_prompt()`
组装完整的 System Prompt + User Prompt，结合知识库上下文和引用规则。

### `insert_citations()`
核心引用校验函数，完整复现 RAGFlow `rag/nlp/search.py` 的逻辑：

1. **代码块保护**: 用 `re.split(r"(```)", answer)` 保护代码块不被篡改
2. **多语言断句**: 用正则 `[^\\|][；。？!！،؛؟۔\n]` 切分句子
3. **噪声过滤**: 过滤长度 < 5 字符的短句
4. **向量化**: 调用 embedding 模型对每个句子做向量化
5. **混合相似度**: `tkweight * term_similarity + vtweight * vector_similarity`（默认 0.1/0.9）
6. **阈值迭代**: 从 0.63 开始，无匹配则 *0.8 递减，最低 0.3
7. **引用注入**: 为每个句子注入 `[ID:x]` 标记，最多 4 个引用/句

## 测试数据

`test_chunks.json` 包含两个测试知识块：

- **[ID:0]**: IBM Content Navigator Administration Guide - Task Manager 不支持的功能
- **[ID:1]**: FileNet Deployment Best Practices - Content Engine 和 Process Engine 部署架构

## 预期输出

### Mock 模式
- LLM 原始回答包含 `[ID:0]`, `[ID:1]`, `[ID:2]`（其中 `[ID:2]` 为幻觉引用）
- 后端校验后，幻觉引用 `[ID:2]` 被消除，有效引用被保留
- 最终输出标准 JSON，包含 `answer` 和 `reference_chunks`

### 真实 API 模式
- 使用配置的 LLM 生成真实回答
- Embedding 向量计算真实相似度
- 阈值策略根据实际相似度动态调整

## RAGFlow 源码对应

| Notebook 函数 | RAGFlow 源码 |
|--------------|-------------|
| `kb_prompt()` | `rag/nlp/search.py` 中的 chunk 格式化逻辑 |
| `citation_prompt()` | `rag/prompts/citation_prompt.md` |
| `build_rag_prompt()` | `rag/prompts/generator.py` 中的 `citation_prompt()` |
| `insert_citations()` | `rag/nlp/search.py` 中的核心方法 |
| 混合相似度 | `tkweight * term_similarity + vtweight * vector_similarity` |
