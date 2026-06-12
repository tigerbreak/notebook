"""Core retrieval engine: metadata extraction → chunking → Qdrant/ES → RRF → Reranker."""

import os
import re
import logging
import jieba
import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Any

from config import (
    EMBEDDING_MODEL, RRF_K, RRF_ALPHA, DENSE_TOP_K, BM25_TOP_K,
    FINAL_TOP_K, RERANK_TOP_K, PARENT_MAX_CHARS, CHILD_MAX_CHARS, CHILD_OVERLAP,
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_VECTOR_DIM,
)
from models import PolicyMeta, Chunk
from data import POLICIES

log = logging.getLogger(__name__)

# ── Global embedder cache ──
_EMBEDDER_CACHE = {}


def _get_embedder() -> "SentenceTransformer":
    """Get or create the shared embedding model."""
    if EMBEDDING_MODEL not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        log.info(f"🔄 加载 Embedding 模型 ({EMBEDDING_MODEL})…")
        _EMBEDDER_CACHE[EMBEDDING_MODEL] = SentenceTransformer(EMBEDDING_MODEL)
        log.info(f"✅ Embedding 模型加载完成 (dim={_EMBEDDER_CACHE[EMBEDDING_MODEL].get_embedding_dimension()})")
    return _EMBEDDER_CACHE[EMBEDDING_MODEL]


# ═══════════════════════════════════════════════════════════════
# 1. Metadata Extraction
# ═══════════════════════════════════════════════════════════════

PATTERNS = {
    "policy_id": re.compile(r"\*\*保单号\*\*[：:]?\s*(\S+)"),
    "insurer": re.compile(r"^(中国[^（\n]+)(?:[（(]|\s*保险)"),
    "applicant": re.compile(r"\*\*投保人\*\*[：:]?\s*(\S+)"),
    "insured": re.compile(r"\*\*被保险人\*\*[：:]?\s*(\S+)"),
    "product": re.compile(r"\*\*产品名称\*\*[：:]?\s*(.+)"),
    "premium": re.compile(r"\*\*保费\*\*[：:]?\s*(\S+)"),
    "eff_date": re.compile(r"\*\*生效日期\*\*[：:]?\s*(\S+)"),
    "beneficiary_line": re.compile(r"(\d+\s*[.、．]?\s*)(\S+)\s*[（(]?(.+?)[）)]?\s*(\d+%)?\s*受益", re.MULTILINE),
    "endorsement_id": re.compile(r"批单号[：:]?\s*(\S+)"),
    "endorsement_date": re.compile(r"(?:申请|生效)日期[：:]?\s*(\S+)"),
    "change_before": re.compile(r"变更前[：:]\s*(.+?)(?=\n\n|\n\*\*)", re.DOTALL),
    "change_after": re.compile(r"变更后[：:]\s*(.+?)(?=\n\n|\n\*\*|\Z)", re.DOTALL),
    "change_reason": re.compile(r"变更原因[：:]?\s*(.+)"),
}


def extract_metadata(policy: dict) -> PolicyMeta:
    text = "\n".join(p["md"] for p in policy["pages"])

    def get(key):
        m = PATTERNS[key].search(text)
        return m.group(1).strip() if m else ""

    pid = get("policy_id")
    insurer = PATTERNS["insurer"].search(text)
    insurer = insurer.group(1).strip() if insurer else policy.get("insurer", "")

    meta = PolicyMeta(id=pid, insurer=insurer, applicant=get("applicant"),
                      insured=get("insured"), product=get("product"),
                      premium=get("premium"), eff_date=get("eff_date"))

    # Beneficiaries (current state)
    ben_lines = PATTERNS["beneficiary_line"].findall(text)
    for b in ben_lines:
        name = b[1]
        rest = b[2] + (" " + b[3] if b[3] else "")
        ratio = b[3] if b[3] else ""
        meta.beneficiaries.append({
            "name": name.strip(),
            "ratio": ratio.strip() if ratio else "100%",
            "relation": rest.strip().rstrip("）)"),
        })

    # Endorsements (batch history)
    for pg in policy["pages"]:
        t = pg["md"]
        if "批单" not in t:
            continue
        eid = PATTERNS["endorsement_id"].search(t)
        before = PATTERNS["change_before"].search(t)
        after = PATTERNS["change_after"].search(t)
        reason = PATTERNS["change_reason"].search(t)
        if eid:
            entry = {"id": eid.group(1), "date": "", "change": "",
                     "before": "", "after": ""}
            for p in policy["pages"]:
                dt = PATTERNS["endorsement_date"].search(p["md"])
                if dt:
                    entry["date"] = dt.group(1)
                    break
            if before:
                entry["before"] = before.group(1).strip().replace("\n", " | ")
            if after:
                entry["after"] = after.group(1).strip().replace("\n", " | ")
            if reason:
                entry["change"] = reason.group(1).strip()
            meta.endorsements.append(entry)

    return meta


# ═══════════════════════════════════════════════════════════════
# 2. Parent-Child Chunking
# ═══════════════════════════════════════════════════════════════

def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Split page markdown into (heading, content) sections."""
    sections = []
    lines = text.split("\n")
    current_heading, current_body = "", []
    for line in lines:
        if line.startswith("**") and line.endswith("**"):
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.strip("*")
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))
    return sections


def chunk_policy(policy: dict) -> List[Chunk]:
    parents, children = [], []
    pid = policy["id"]

    for page in policy["pages"]:
        pn = page["page"]
        sections = _split_into_sections(page["md"])
        for heading, body in sections:
            combined = f"{heading}\n{body}" if heading else body
            if not combined.strip():
                continue
            # Parent chunks (page-level or section-level)
            for start in range(0, len(combined), PARENT_MAX_CHARS):
                parent_text = combined[start:start + PARENT_MAX_CHARS]
                pc = Chunk(cid=f"p-{pid}-p{pn}-{len(parents)}", pid=pid,
                           page=pn, type="parent", heading=heading, text=parent_text)
                parents.append(pc)
            # Child chunks (smaller, for retrieval)
            cs_text = body if heading else combined
            cs_text = cs_text.strip()
            for start in range(0, len(cs_text), CHILD_MAX_CHARS - CHILD_OVERLAP):
                chunk_text = cs_text[start:start + CHILD_MAX_CHARS]
                if len(chunk_text) < 20 and start > 0:
                    continue
                cc = Chunk(cid=f"c-{pid}-p{pn}-{len(children)}", pid=pid,
                           page=pn, type="child", parent_id=parents[-1].cid if parents else "",
                           heading=heading, text=chunk_text.strip())
                children.append(cc)

    return parents + children


# ═══════════════════════════════════════════════════════════════
# 3. Tokenization (module-level for reuse)
# ═══════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """Tokenize Chinese text using jieba."""
    tokens = []
    for w in jieba.cut(text):
        w = w.strip()
        if w and w not in ("", " ", "\n", "|", "---", "**:"):
            tokens.append(w.lower())
    return tokens


# ═══════════════════════════════════════════════════════════════
# 4. PostgreSQL (pgvector) Index
# ═══════════════════════════════════════════════════════════════

TABLE_NAME = "policy_chunks"


class PGVectorIndex:
    """Vector + full-text index backed by PostgreSQL with pgvector extension."""

    def __init__(self):
        self.conn = None
        self._ready = False

    def connect(self) -> bool:
        """Connect to PostgreSQL, enable pgvector, create schema."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASSWORD,
            )
            self.conn.autocommit = True
            cur = self.conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._ensure_schema()
            self._ready = True
            log.info(f"✅ PostgreSQL 连接成功 ({PG_HOST}:{PG_PORT})")
            return True
        except Exception as e:
            log.warning(f"⚠️ PostgreSQL 连接失败（将使用内存检索）: {e}")
            self._ready = False
            return False

    def _ensure_schema(self):
        """Create table, indexes, and functions if not exist."""
        cur = self.conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                chunk_id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                heading TEXT DEFAULT '',
                text TEXT NOT NULL,
                chunk_type TEXT DEFAULT 'child',
                parent_id TEXT DEFAULT '',
                embedding vector({PG_VECTOR_DIM}),
                tokens TEXT[] DEFAULT '{{}}'
            )
        """)
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_policy ON {TABLE_NAME} (policy_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_embedding ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_tokens ON {TABLE_NAME} USING gin (tokens)")
        log.info(f"📦 PostgreSQL schema 已就绪 ({TABLE_NAME}, dim={PG_VECTOR_DIM})")

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Insert chunks with vectors and tokens."""
        if not self._ready:
            return
        try:
            import psycopg2.extras
            cur = self.conn.cursor()
            values = []
            for ch, emb in zip(chunks, embeddings):
                t = tokenize(ch.text)
                values.append((ch.cid, ch.pid, ch.page, ch.heading, ch.text,
                               ch.type, ch.parent_id, emb, t))
            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO {TABLE_NAME}
                (chunk_id, policy_id, page_number, heading, text, chunk_type, parent_id, embedding, tokens)
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    tokens = EXCLUDED.tokens
                """,
                values,
                template="(%s, %s, %s, %s, %s, %s, %s, %s::vector, %s::text[])",
            )
            log.info(f"📤 PostgreSQL 写入 {len(values)} 个 chunks")
        except Exception as e:
            log.warning(f"⚠️ PostgreSQL 写入失败: {e}")

    def vector_search(self, vector: List[float], top_k: int = DENSE_TOP_K,
                      policy_id: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        """Cosine similarity search via pgvector <=> operator."""
        if not self._ready:
            return []
        try:
            cur = self.conn.cursor()
            if policy_id:
                cur.execute(f"""
                    SELECT chunk_id, policy_id, page_number, heading, text, chunk_type, parent_id,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {TABLE_NAME}
                    WHERE chunk_type = 'child' AND policy_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vector, policy_id, vector, top_k))
            else:
                cur.execute(f"""
                    SELECT chunk_id, policy_id, page_number, heading, text, chunk_type, parent_id,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {TABLE_NAME}
                    WHERE chunk_type = 'child'
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vector, vector, top_k))
            results = []
            for row in cur.fetchall():
                ch = Chunk(cid=row[0], pid=row[1], page=row[2],
                           heading=row[3], text=row[4], type=row[5], parent_id=row[6])
                results.append((ch, float(row[7])))
            return results
        except Exception as e:
            log.warning(f"⚠️ PostgreSQL 向量检索失败: {e}")
            return []

    def fulltext_search(self, query: str, top_k: int = BM25_TOP_K,
                        policy_id: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        """Full-text search using jieba token array overlap."""
        if not self._ready:
            return []
        try:
            query_tokens = tokenize(query)
            if not query_tokens:
                return []
            cur = self.conn.cursor()
            if policy_id:
                cur.execute(f"""
                    SELECT chunk_id, policy_id, page_number, heading, text, chunk_type, parent_id,
                           cardinality(tokens::text[] & %s::text[]) AS score
                    FROM {TABLE_NAME}
                    WHERE chunk_type = 'child'
                      AND policy_id = %s
                      AND tokens && %s::text[]
                    ORDER BY score DESC
                    LIMIT %s
                """, (query_tokens, policy_id, query_tokens, top_k))
            else:
                cur.execute(f"""
                    SELECT chunk_id, policy_id, page_number, heading, text, chunk_type, parent_id,
                           cardinality(tokens::text[] & %s::text[]) AS score
                    FROM {TABLE_NAME}
                    WHERE chunk_type = 'child'
                      AND tokens && %s::text[]
                    ORDER BY score DESC
                    LIMIT %s
                """, (query_tokens, query_tokens, top_k))
            results = []
            for row in cur.fetchall():
                ch = Chunk(cid=row[0], pid=row[1], page=row[2],
                           heading=row[3], text=row[4], type=row[5], parent_id=row[6])
                results.append((ch, float(row[7])))
            return results
        except Exception as e:
            log.warning(f"⚠️ PostgreSQL 全文检索失败: {e}")
            return []

    @property
    def is_ready(self) -> bool:
        return self._ready


# ═══════════════════════════════════════════════════════════════
# 4. Retrieval Engine (orchestrator)
# ═══════════════════════════════════════════════════════════════

class RetrievalEngine:
    """Orchestrates hybrid search across Qdrant/ES/memory + Reranker."""

    def __init__(self):
        self.policies = POLICIES
        self.metas: Dict[str, PolicyMeta] = {}
        self.chunks: List[Chunk] = []
        self.children: List[Chunk] = []
        self.embedder = None
        # Fallback in-memory indexes
        self.dense_matrix: Optional[np.ndarray] = None
        self.bm25 = None
        self.tokenized_chunks: List[List[str]] = []
        # External PostgreSQL + pgvector index
        self.pg = PGVectorIndex()
        self._reranker = None
        self._ready = False

    def build(self):
        """Build indexes from mock data, connecting to PostgreSQL if available."""
        # --- Metadata & chunks ---
        all_chunks = []
        for pol in self.policies:
            pid = pol["id"]
            self.metas[pid] = extract_metadata(pol)
            all_chunks.extend(chunk_policy(pol))
        self.chunks = all_chunks
        self.children = [c for c in all_chunks if c.type == "child"]
        log.info(f"📄 解析完成: {len(self.metas)} 份保单, {len(self.chunks)} chunks")

        # --- Embedding model ---
        log.info(f"🔄 加载 Embedding 模型 ({EMBEDDING_MODEL})…")
        self._load_embedder()

        # --- Connect PostgreSQL ---
        pg_ok = self.pg.connect()

        # --- Compute embeddings ---
        ctexts = [c.text for c in self.children]
        log.info(f"📐 编码 {len(ctexts)} 个子块向量…")
        embeddings = self.embedder.encode(ctexts, show_progress_bar=False)
        log.info(f"✅ 向量矩阵 shape: {embeddings.shape}")

        # --- Write to PostgreSQL ---
        if pg_ok:
            self.pg.add_chunks(self.children, embeddings.tolist())

        # --- Always build in-memory fallback ---
        self.dense_matrix = embeddings
        self.tokenized_chunks = [tokenize(c.text) for c in self.children]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        vocab_size = len(set(w for t in self.tokenized_chunks for w in t))
        log.info(f"✅ 内存 BM25 索引构建完成（词表大小: ~{vocab_size})")

        self._ready = True
        return self

    def _load_embedder(self):
        """Load the embedding model (uses module-level cache to avoid hangs)."""
        self.embedder = _get_embedder()

    @property
    def active_db(self) -> str:
        if self.pg.is_ready:
            return "pgvector"
        return "memory"

    @property
    def active_fts(self) -> str:
        if self.pg.is_ready:
            return "pgvector"
        return "bm25"

    @property
    def reranker_status(self) -> str:
        if self._reranker and self._reranker.is_ready:
            return "active"
        return "inactive"

    # ── Dense search ──

    def _dense_search(self, query: str, top_k: int = DENSE_TOP_K,
                      policy_id: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        qv = self.embedder.encode([query])[0].tolist()

        # Try PostgreSQL pgvector first
        if self.pg.is_ready:
            results = self.pg.vector_search(qv, top_k, policy_id)
            if results:
                return results

        # Fallback to in-memory cosine similarity
        qv_np = np.array(qv)
        scores = np.dot(self.dense_matrix, qv_np) / (
            np.linalg.norm(self.dense_matrix, axis=1) * np.linalg.norm(qv_np) + 1e-10
        )
        if policy_id:
            mask = np.array([c.pid == policy_id for c in self.children])
            scores[~mask] = -1
        top = np.argsort(scores)[-top_k:][::-1]
        return [(self.children[i], float(scores[i])) for i in top if scores[i] > 0]

    # ── BM25 / Full-text search ──

    def _bm25_search(self, query: str, top_k: int = BM25_TOP_K,
                     policy_id: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        # Try PostgreSQL full-text first
        if self.pg.is_ready:
            results = self.pg.fulltext_search(query, top_k, policy_id)
            if results:
                return results

        # Fallback to in-memory rank_bm25
        qt = tokenize(query)
        scores = self.bm25.get_scores(qt)
        if policy_id:
            for i, c in enumerate(self.children):
                if c.pid != policy_id:
                    scores[i] = -1
        top = np.argsort(scores)[-top_k:][::-1]
        return [(self.children[i], float(scores[i])) for i in top if scores[i] > 0]

    # ── Hybrid search ──

    def hybrid_search(self, query: str, policy_id: Optional[str] = None,
                      top_k: int = FINAL_TOP_K) -> List[Tuple[Chunk, float, str]]:
        """RRF fusion of dense + BM25, optional reranker."""
        dense_res = self._dense_search(query, DENSE_TOP_K, policy_id)
        bm25_res = self._bm25_search(query, BM25_TOP_K, policy_id)

        scores = defaultdict(float)
        for rank, (ch, _) in enumerate(dense_res):
            scores[ch.cid] += RRF_ALPHA / (rank + RRF_K)
        for rank, (ch, _) in enumerate(bm25_res):
            scores[ch.cid] += (1 - RRF_ALPHA) / (rank + RRF_K)

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        result = [(self._chunk_by_id(cid), sc, "hybrid") for cid, sc in ranked if self._chunk_by_id(cid)]

        # Reranker
        if self._ensure_reranker():
            result = self._reranker.rerank(query, result, top_k=RERANK_TOP_K)

        return result[:top_k]

    def _ensure_reranker(self) -> bool:
        """Lazy-load the reranker model."""
        if self._reranker is None:
            from reranker import Reranker
            self._reranker = Reranker()
            self._reranker.load()
        return self._reranker.is_ready

    # ── Helpers ──

    def _chunk_by_id(self, cid: str) -> Optional[Chunk]:
        for c in self.chunks:
            if c.cid == cid:
                return c
        return None

    def get_context(self, chunks: List[Tuple[Chunk, float, str]],
                    max_chars: int = 3000) -> str:
        parts = []
        for ch, sc, src in chunks:
            parts.append(
                f"[保单 {ch.pid} | 第 {ch.page} 页 | 段落: {ch.heading or '(正文)'}]\n"
                f"{ch.text}"
            )
        ctx = "\n\n".join(parts)
        return ctx[:max_chars]

    def get_metadata_str(self, policy_id: str) -> str:
        m = self.metas.get(policy_id)
        return m.to_prompt() if m else ""

    def get_policy_detail(self, policy_id: str) -> Optional[Dict[str, Any]]:
        meta = self.metas.get(policy_id)
        if not meta:
            return None
        policy_chunks = [c for c in self.chunks if c.pid == policy_id]
        pages = sorted({c.page for c in policy_chunks})
        return {
            "policy_id": policy_id,
            "metadata": meta.to_dict(),
            "page_count": len(pages),
            "chunk_count": len(policy_chunks),
        }

    @property
    def is_ready(self) -> bool:
        return self._ready


# Global singleton
engine = RetrievalEngine()
