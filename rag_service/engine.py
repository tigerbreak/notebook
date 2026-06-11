"""Core retrieval engine: metadata extraction → chunking → dense/BM25 → RRF fusion."""

import re
import jieba
import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, RRF_K, RRF_ALPHA, DENSE_TOP_K, BM25_TOP_K,
    FINAL_TOP_K, PARENT_MAX_CHARS, CHILD_MAX_CHARS, CHILD_OVERLAP,
)
from models import PolicyMeta, Chunk
from data import POLICIES


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
            last = len(children)
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
# 3. Retrieval Indexes
# ═══════════════════════════════════════════════════════════════

class RetrievalEngine:
    """Holds all indexes and provides hybrid search."""

    def __init__(self):
        self.policies = POLICIES
        self.metas: Dict[str, PolicyMeta] = {}
        self.chunks: List[Chunk] = []
        self.children: List[Chunk] = []
        self.embedder: SentenceTransformer = None
        self.dense_matrix: np.ndarray = None
        self.bm25: BM25Okapi = None
        self.tokenized: List[List[str]] = []
        self._ready = False

    def build(self):
        """Build all indexes from mock data."""
        # Metadata & chunks
        all_chunks = []
        for pol in self.policies:
            pid = pol["id"]
            self.metas[pid] = extract_metadata(pol)
            all_chunks.extend(chunk_policy(pol))
        self.chunks = all_chunks
        self.children = [c for c in all_chunks if c.type == "child"]

        # Embedding
        print(f"🔄 加载 Embedding 模型 ({EMBEDDING_MODEL})…")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        ctexts = [c.text for c in self.children]
        print(f"📐 编码 {len(ctexts)} 个子块向量…")
        self.dense_matrix = self.embedder.encode(ctexts, show_progress_bar=False)
        print(f"✅ 向量矩阵 shape: {self.dense_matrix.shape}")

        # BM25
        self.tokenized = [self._tokenize(c.text) for c in self.children]
        self.bm25 = BM25Okapi(self.tokenized)
        print(f"✅ BM25 索引构建完成（词表大小: ~{len(set(w for t in self.tokenized for w in t))})")

        self._ready = True
        return self

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = []
        for w in jieba.cut(text):
            w = w.strip()
            if w and w not in ("", " ", "\n", "|", "---", "**:"):
                tokens.append(w.lower())
        return tokens

    def _dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> List[Tuple[Chunk, float]]:
        qv = self.embedder.encode([query])[0]
        scores = np.dot(self.dense_matrix, qv) / (
            np.linalg.norm(self.dense_matrix, axis=1) * np.linalg.norm(qv) + 1e-10
        )
        top = np.argsort(scores)[-top_k:][::-1]
        return [(self.children[i], float(scores[i])) for i in top]

    def _bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> List[Tuple[Chunk, float]]:
        qt = self._tokenize(query)
        scores = self.bm25.get_scores(qt)
        top = np.argsort(scores)[-top_k:][::-1]
        return [(self.children[i], float(scores[i])) for i in top if scores[i] > 0]

    def hybrid_search(self, query: str, policy_id: Optional[str] = None,
                      top_k: int = FINAL_TOP_K) -> List[Tuple[Chunk, float, str]]:
        """RRF fusion of dense + BM25, with optional metadata filter."""
        dense_res = self._dense_search(query, DENSE_TOP_K)
        bm25_res = self._bm25_search(query, BM25_TOP_K)

        scores = defaultdict(float)
        for rank, (ch, _) in enumerate(dense_res):
            scores[ch.cid] += RRF_ALPHA / (rank + RRF_K)
        for rank, (ch, _) in enumerate(bm25_res):
            scores[ch.cid] += (1 - RRF_ALPHA) / (rank + RRF_K)

        # Metadata filter
        if policy_id:
            scores = {cid: sc for cid, sc in scores.items()
                      if self._chunk_by_id(cid).pid == policy_id}

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        result = []
        for cid, sc in ranked:
            ch = self._chunk_by_id(cid)
            source = "hybrid"
            result.append((ch, sc, source))
        return result

    def _chunk_by_id(self, cid: str) -> Optional[Chunk]:
        for c in self.chunks:
            if c.cid == cid:
                return c
        return None

    def get_context(self, chunks: List[Tuple[Chunk, float, str]],
                    max_chars: int = 3000) -> str:
        parts = []
        for ch, sc, src in chunks:
            meta = self.metas.get(ch.pid)
            parts.append(
                f"[保单 {ch.pid} | 第 {ch.page} 页 | 段落: {ch.heading or '(正文)'}]\n"
                f"{ch.text}"
            )
        ctx = "\n\n".join(parts)
        return ctx[:max_chars]

    def get_metadata_str(self, policy_id: str) -> str:
        m = self.metas.get(policy_id)
        return m.to_prompt() if m else ""

    @property
    def is_ready(self) -> bool:
        return self._ready


# Global singleton
engine = RetrievalEngine()
