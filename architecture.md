# 保险保单受益人变更检索系统 — 架构方案

## 业务场景

保险公司持有海量多页 TIF 格式的历史保单，当发生受益人变更时，理赔人员需要快速定位：

1. **哪份保单**发生了变更？
2. **变更发生在哪一页、哪一段**？
3. **变更前后的具体内容是什么**？

## 系统架构

```
                          ┌──────────────┐
                          │  多页 TIF保单 │
                          └──────┬───────┘
                                 ▼
                    ┌──────────────────────┐
                    │  OSS 对象存储 + 触发  │
                    └──────┬───────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ① 解析层 (ETL)                                                │
│     OCR+Markdown  →  元数据提取  →  父子分块 (Parent-Child)    │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ② 索引层                                                      │
│     ┌──────────────────┐      ┌──────────────────┐             │
│     │  向量索引 (Dense) │      │  全文索引 (BM25)  │             │
│     │  Sentence-BERT   │      │  jieba + BM25    │             │
│     └────────┬─────────┘      └────────┬─────────┘             │
└──────────────┼─────────────────────────┼────────────────────────┘
               └───────────┬─────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ③ 检索与重排                                                  │
│     RRF 融合 + Metadata 过滤 + 重排序 → Top-5 精准片段         │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ④ 生成 (LLM)                                                  │
│     提示词 + 上下文 → 结构化答案 (变更前后对比 + 位置引用)     │
└─────────────────────────────────────────────────────────────────┘
```

## 流程说明

| 阶段 | 模块 | 说明 |
|------|------|------|
| **① 解析层** | OCR + Markdown | TIF 保单经 OCR 识别为结构化 Markdown 文本 |
| | 元数据提取 | 正则提取保单号、投保人、受益人、批单历史等结构化字段 |
| | 父子分块 | Parent-Child Chunking：子块用于检索，父块用于上下文 |
| **② 索引层** | 向量索引 (Dense) | Sentence-BERT 编码为稠密向量，余弦相似度检索 |
| | 全文索引 (BM25) | jieba 分词 + BM25 算法，精确匹配关键词 |
| **③ 检索层** | RRF 融合 | Reciprocal Rank Fusion 融合两路检索结果 |
| | Metadata 过滤 | 按保单号精准过滤，排除其他保单干扰 |
| | 重排序 | 精排模型对 Top-N 结果重打分 |
| **④ 生成层** | Prompt 构造 | 上下文 + 元数据 + 问题 → 结构化 Prompt |
| | LLM 生成 | 大模型输出变更前后对比、所在页码等结构化答案 |

## 关键技术选型

| 模块 | PoC 方案 | 生产推荐 |
|------|----------|----------|
| OCR | 模拟数据 | PP-OCRv4 / LayoutLMv3 |
| 向量模型 | all-MiniLM-L6-v2 | BAAI/bge-large-zh-v1.5 |
| 向量数据库 | 内存数组 | Milvus / Qdrant / pgvector |
| 全文索引 | rank_bm25 | Elasticsearch 8.x + IK 分词 |
| 精排模型 | — | BAAI/bge-reranker-v2-m3 |
| 大语言模型 | Mock LLM | DeepSeek / Qwen / GPT-4o |

## 检索策略

采用混合检索 + Metadata 过滤的方式：

```
Score(chunk) = α × RRF_dense(chunk) + (1-α) × RRF_bm25(chunk)
```

- 向量检索捕捉语义相关性（如"受益比例变更"）
- BM25 检索精确匹配关键词（如"张美玲 60%"）
- RRF 融合两路排序，过滤阶段按保单号精确筛选

## PoC 验证结果

- **测试用例**: 3 个（含变更 / 多次变更 / 无变更保单）
- **页面命中率**: 100%（3/3）
- **关键词召回率**: 100%（全命中）

---

# 技术实现方案

## 一、前置条件清单

在实施前，需要确认以下条件是否具备：

### 1.1 LLM 服务（已就绪 ✅）

| 项目 | 值 |
|------|-----|
| Provider | DeepSeek |
| Base URL | `https://api.deepseek.com` |
| Model | `deepseek-v4-flash` |
| API Key | 已提供 |
| SDK | `pip install openai`（兼容 OpenAI SDK） |

### 1.2 基础设施（推荐配置）

| 组件 | 方案选项 | 推荐理由 |
|------|----------|----------|
| **向量数据库** | Qdrant（Docker） | 支持标量过滤 + 向量混合查询，单机部署简单 |
| **全文索引** | Elasticsearch 8.x（Docker） | IK 中文分词插件成熟 |
| **应用框架** | FastAPI + uvicorn | 异步支持好，天然对接 Pydantic |
| **前端** | Streamlit | 快速搭建检索 Demo 界面 |
| **任务队列** | Celery + Redis（可选） | 异步处理大批量 TIF 入库 |

### 1.3 依赖清单（Python）

```
# 核心依赖
openai                    # DeepSeek / LLM 调用
fastapi + uvicorn         # API 服务
qdrant-client             # 向量数据库客户端
elasticsearch             # 全文检索客户端
sentence-transformers     # 向量编码
jieba + rank_bm25         # 中文分词 + BM25（备选）

# 辅助
pydantic                  # 数据模型
streamlit                 # 前端 Demo
celery + redis            # 异步任务（可选）
```

---

## 二、架构分层实现思路

### 2.1 接入层（API Gateway）

```
POST /api/v1/search          # 检索接口
POST /api/v1/answer          # 检索 + LLM 问答
POST /api/v1/policies/upload # TIF 保单上传
GET  /api/v1/policies/{id}   # 保单详情
```

**实现要点：**
- FastAPI 异步路由，请求/响应模型用 Pydantic 定义
- 上传接口返回 `task_id`，异步处理通过 WebSocket 或轮询获取状态
- 检索接口设计：`question + policy_id（可选）→ chunks + llm_answer`

### 2.2 ETL 解析层

**流程图：**
```
TIF 上传
  │
  ▼
┌──────────────┐
│ OCR 识别      │  ← PP-OCR 或 Tesseract（需安装）
│ → Markdown   │     中文保单推荐 PP-OCRv4
└──────┬───────┘
       ▼
┌──────────────┐
│ 元数据提取    │  ← 正则表达式 + 规则引擎（已有 PoC 代码）
│ → 结构化字段  │
└──────┬───────┘
       ▼
┌──────────────┐
│ 父子分块      │  ← Parent-Child Chunking（已有 PoC 代码）
│ → Chunks     │
└──────┬───────┘
       ▼
┌──────────────┐
│ 双通道写入    │
└──────┬───────┘
       │
  ┌────┴────┐
  ▼         ▼
Qdrant    Elasticsearch
(向量)    (全文索引)
```

**OCR 方案对比：**

| 方案 | 中文精度 | 部署难度 | 是否需 GPU |
|------|----------|----------|-----------|
| **PP-OCRv4**（PaddleOCR） | ⭐⭐⭐⭐⭐ | 中（需安装 Paddle） | 可选 |
| **Tesseract 5 + chi_sim** | ⭐⭐⭐ | 低（apt install） | 否 |
| **DocTR** | ⭐⭐⭐⭐ | 中（pip 安装） | 可选 |
| **模拟数据（PoC）** | — | 低 | 否 |

> 💡 **建议**：PoC 阶段继续使用模拟 Markdown 数据；进入开发阶段后集成 PP-OCRv4。

### 2.3 索引层

#### 向量索引（Qdrant）

```
Collection: policy_chunks
  │
  ├─ payload: { policy_id, page_number, heading, chunk_id, parent_id }
  ├─ vector:  384-dim (all-MiniLM-L6-v2) / 1024-dim (bge-large-zh)
  └─ index:   HNSW (默认配置)
```

**关键设计决策：**

| 决策点 | 选项 | 推荐 |
|--------|------|------|
| 向量维度 | 384 / 768 / 1024 | PoC 用 384；生产用 1024（bge-large-zh） |
| Embedding 模型 | all-MiniLM / bge / m3e | 中文场景推荐 `BAAI/bge-large-zh-v1.5` |
| 模型下载时机 | 启动时加载 | 建议单独初始化为全局变量 |
| 批量大小 | 32 / 64 / 128 | 取决于 CPU 内存，推荐 64 |

**Qdrant 启动命令（Docker）：**
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

#### 全文索引（Elasticsearch）

```
Index: policy_chunks
  │
  ├─ fields: text (text, ik_smart analyzer)
  │          policy_id (keyword)
  │          page_number (integer)
  │          heading (text)
  └─ query:  bool (must: match(text), filter: policy_id)
```

**IK 分词器配置：**
```json
{
  "settings": {
    "analysis": {
      "analyzer": { "ik_smart": { "type": "custom", "tokenizer": "ik_smart" } }
    }
  }
}
```

**ES 启动命令（Docker）：**
```bash
docker run -d --name es -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.15.0

# 安装 IK 分词插件
docker exec es elasticsearch-plugin install \
  https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.15.0/elasticsearch-analysis-ik-8.15.0.zip
```

### 2.4 检索层

#### 混合检索 Pipeline

```
用户查询
  │
  ├──→ Qdrant 向量检索（Top-100）
  │      Query → Embedding → ANN Search → (chunk_id, score)
  │
  ├──→ ES 全文检索（Top-100）
  │      Query → IK 分词 → BM25 → (chunk_id, score)
  │
  └──→ RRF 融合
         score = α · rank⁻¹(qdrant) + (1-α) · rank⁻¹(es)
         │
         ▼
       Metadata 过滤
         │ (policy_id 精确匹配)
         ▼
       ╔═══════════════════╗
       ║ 精排 (Reranker)   ║  ← 可选：bge-reranker-v2-m3
       ║ Cross-encoder     ║     对 Top-20 两两打分
       ╚═══════════════════╝
         │
         ▼
       Top-5 最终结果
```

**RRF 融合伪代码：**
```python
def hybrid_search(query, policy_id=None, alpha=0.5):
    # 1. 两路检索
    qdrant_hits = qdrant_client.search(
        collection_name="policy_chunks",
        query_vector=encode(query),
        limit=100
    )
    es_hits = es_client.search(
        index="policy_chunks",
        body={"query": {"match": {"text": query}}},
        size=100
    )

    # 2. RRF 分数融合
    K = 60
    scores = defaultdict(float)
    for rank, hit in enumerate(qdrant_hits):
        scores[hit.id] += alpha / (rank + K)
    for rank, hit in enumerate(es_hits["hits"]["hits"]):
        scores[hit["_id"]] += (1 - alpha) / (rank + K)

    # 3. Metadata 过滤
    if policy_id:
        scores = {k: v for k, v in scores.items()
                  if get_chunk(k).policy_id == policy_id}

    # 4. 精排（可选）
    top_n = sorted(scores.items(), key=lambda x: -x[1])[:20]
    reranked = reranker_model.rerank(query, [get_chunk(k).text for k, _ in top_n])
    # 5. 返回 Top-5
    return reranked[:5]
```

**精排模型接入：**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank(self, query, texts):
        pairs = [[query, text] for text in texts]
        inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                return_tensors="pt", max_length=512)
        scores = self.model(**inputs).logits.squeeze(-1).tolist()
        return sorted(zip(texts, scores), key=lambda x: -x[1])
```

### 2.5 生成层（LLM）

#### DeepSeek 集成

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-bf95ba0f0f354054bdc79e3101a971ae",
    base_url="https://api.deepseek.com"
)

def llm_answer(context, metadata, question):
    prompt = f"""你是一位专业的保险理赔分析师。请根据以下保单上下文，回答用户的问题。

## 上下文（保单片段）
{context}

## 保单元数据
{metadata}

## 用户问题
{question}

## 回答要求
1. 如果上下文明确包含答案，请引用原文并注明页码。
2. 涉及变更时，必须列出变更前后对比表格。
3. 如果信息不足，请如实说明。
4. 使用中文、列表格式回答。

## 答案"""

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000
    )
    return response.choices[0].message.content
```

**Prompt Engineering 要点：**

| 维度 | 建议 |
|------|------|
| **temperature** | 0.1（业务场景需确定性，不宜高） |
| **max_tokens** | 2000（答案可能包含表格） |
| **system prompt** | 可增加角色设定，强化"引用原文"约束 |
| **few-shot** | 可在 system 层加 1-2 个示例 |
| **输出格式** | 要求结构化（表格 + 列表），便于前端渲染 |

### 2.6 异步任务编排

对于 TIF 上传 → OCR → 入库的耗时长流程，建议引入异步任务队列：

```
TIF 上传 OSS
  │
  ▼
┌──────────────────┐
│ 生成 task_id      │
│ 返回 (202 Accepted)
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Celery Worker    │
│                   │
│ 1. 从 OSS 下载 TIF│
│ 2. OCR → Markdown│
│ 3. 元数据提取     │
│ 4. 父子分块       │
│ 5. 写入 Qdrant    │
│ 6. 写入 ES        │
│ 7. 更新任务状态   │
└───────────────────┘
```

**技术选型：**
- **Celery + Redis**：最成熟，但配置较重
- **FastAPI BackgroundTasks**：简单，但不支持分布式
- **APScheduler**：适合定时调度

> 💡 PoC 阶段直接用同步流程即可，生产再引入异步。

---

## 三、系统集成总览

```python
# ─── 核心数据结构 ────────────────────────────────────────────
@dataclass
class Chunk:
    id: str
    policy_id: str
    page_number: int
    heading: str
    text: str
    embedding: List[float]  # 可选，Qdrant 内部管理

@dataclass
class PolicyMeta:
    policy_id: str
    insurer: str
    applicant: str
    insured: str
    product: str
    beneficiaries: List[Dict]
    endorsements: List[Dict]

# ─── 系统接口 ────────────────────────────────────────────────
class OCRPipe:
    def tif_to_markdown(tif_path: str) -> List[Dict]: ...

class MetadataExtractor:
    def extract(pages: List[Dict]) -> PolicyMeta: ...

class Chunker:
    def chunk(pages: List[Dict]) -> List[Chunk]: ...

class VectorIndex:
    def add(chunks: List[Chunk]): ...
    def search(query: str, top_k: int) -> List[Tuple[Chunk, float]]: ...

class FullTextIndex:
    def add(chunks: List[Chunk]): ...
    def search(query: str, top_k: int) -> List[Tuple[Chunk, float]]: ...

class HybridRetriever:
    def retrieve(query: str, policy_id: Optional[str]) -> List[Chunk]: ...

class LLMGenerator:
    def answer(question: str, context: str, metadata: str) -> str: ...
```

---

## 四、部署架构

```
┌──────────────────────────────────────────────────────────┐
│  Docker Compose                                          │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ FastAPI   │  │ Qdrant   │  │ ES       │  │ Streamlit│ │
│  │ :8000     │  │ :6333    │  │ :9200    │  │ :8501    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│        │              │             │                     │
│        └──────────────┴─────────────┘                     │
│                        │                                  │
│                   ┌────┴────┐                             │
│                   │  Redis  │  (可选)                      │
│                   └─────────┘                             │
└──────────────────────────────────────────────────────────┘
```

**docker-compose.yml 核心结构：**
```yaml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [qdrant, es]

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: ["./qdrant_storage:/qdrant/storage"]

  es:
    image: elasticsearch:8.15.0
    ports: ["9200:9200"]
    environment: ["discovery.type=single-node", "xpack.security.enabled=false"]

  streamlit:
    build: ./frontend
    ports: ["8501:8501"]
    depends_on: [api]
```

---

## 五、实施路线图

```
Phase 1 ─ 服务化（1-2 天）
  ├── FastAPI 封装现有 PoC 逻辑
  ├── 对接 DeepSeek API
  └── Streamlit 检索界面

Phase 2 ─ 持久化（2-3 天）
  ├── Docker 部署 Qdrant + ES
  ├── 迁移内存索引到 Qdrant
  ├── 迁移 BM25 到 ES + IK
  └── 验证混合检索一致性

Phase 3 ─ 精排增强（1 天）
  ├── 接入 bge-reranker-v2-m3
  └── Rerank 前后效果对比

Phase 4 ─ OCR 集成（2-3 天）
  ├── 安装 PP-OCR 或 Tesseract
  ├── 接入真实 TIF 识别管道
  └── 端到端验证真实保单

Phase 5 ─ 生产加固（持续）
  ├── Celery 异步任务
  ├── 监控 + 日志
  ├── 评估集自动化
  └── 性能调优
```

---

## 六、总结：还需要什么？

| 组件 | 当前状态 | 是否必须 | 替代方案 |
|------|----------|----------|----------|
| DeepSeek API Key | ✅ 已提供 | 是 | — |
| Docker | ✅ 可用 | 推荐 | 可用 SQLite 替代 ES/Qdrant |
| Qdrant / ES | ❌ 需启动 | 生产必须 | PoC 可用内存索引 |
| OCR 引擎 | ❌ 需安装 | 生产必须 | PoC 可用模拟数据 |
| 前端界面 | ❌ 需开发 | 可选 | 直接用 API + curl 测试 |
| 真实 TIF 数据 | ❌ 需提供 | 验证必须 | PoC 用模拟数据 |

**从 Phase 1 开始**，你就能看到一个可运行的**API 检索服务**。每完成一个 Phase，系统就多一层生产化能力。
