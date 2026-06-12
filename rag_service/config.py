"""Configuration for the Insurance Policy RAG Service."""

import os

# ── DeepSeek API ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-v4-flash"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2000

# ── Embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80 MB, 384-dim

# ── Retrieval ─────────────────────────────────────────────────
RRF_K = 60            # RRF 常数
RRF_ALPHA = 0.5       # Dense vs BM25 权重
DENSE_TOP_K = 50      # 向量检索初选数
BM25_TOP_K = 50       # BM25 初选数
RERANK_TOP_K = 20     # 送入精排的候选数
FINAL_TOP_K = 5       # 最终返回数

# ── Chunking ──────────────────────────────────────────────────
PARENT_MAX_CHARS = 1500
CHILD_MAX_CHARS = 300
CHILD_OVERLAP = 30

# ── PostgreSQL (pgvector) ─────────────────────────────────────
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DB = os.environ.get("PG_DB", "rag_policies")
PG_USER = os.environ.get("PG_USER", "rag")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "rag_pass")
PG_VECTOR_DIM = 384  # all-MiniLM-L6-v2

# ── Reranker ──────────────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cpu"

# ── Service ───────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
API_PREFIX = "/api/v1"
