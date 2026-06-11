"""Configuration for the Insurance Policy RAG Service."""

# ── DeepSeek API ──────────────────────────────────────────────
DEEPSEEK_API_KEY = "os.environ.get("DEEPSEEK_API_KEY", "")"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2000

# ── Embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80 MB

# ── Retrieval ─────────────────────────────────────────────────
RRF_K = 60            # RRF 常数
RRF_ALPHA = 0.5       # Dense vs BM25 权重
DENSE_TOP_K = 20      # 向量检索初选数
BM25_TOP_K = 20       # BM25 初选数
FINAL_TOP_K = 5       # 最终返回数
RERANK_TOP_K = 20     # 送入精排的候选数

# ── Chunking ──────────────────────────────────────────────────
PARENT_MAX_CHARS = 1500
CHILD_MAX_CHARS = 300
CHILD_OVERLAP = 30

# ── Service ───────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
