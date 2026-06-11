"""Data models for the Insurance Policy RAG Service."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel


# ── Core domain models ────────────────────────────────────────

@dataclass
class PolicyMeta:
    """Extracted metadata from a policy."""
    id: str
    insurer: str
    applicant: str = ""
    insured: str = ""
    product: str = ""
    premium: str = ""
    eff_date: str = ""
    beneficiaries: List[Dict] = field(default_factory=list)
    endorsements: List[Dict] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [f"保单号：{self.id}", f"保险公司：{self.insurer}",
                 f"投保人：{self.applicant}", f"被保险人：{self.insured}",
                 f"产品名称：{self.product}", f"保费：{self.premium}",
                 f"生效日期：{self.eff_date}"]
        if self.beneficiaries:
            lines.append("当前受益人：")
            for b in self.beneficiaries:
                lines.append(f"  - {b.get('name','?')} 比例 {b.get('ratio','?')} 与被保人关系 {b.get('relation','?')}")
        if self.endorsements:
            lines.append("历史批单：")
            for e in self.endorsements:
                lines.extend([f"  - 批单号 {e.get('id','?')} ({e.get('date','?')})",
                              f"    变更内容：{e.get('change','?')}"])
        return "\n".join(lines)


@dataclass
class Chunk:
    """A chunk of policy text."""
    cid: str
    pid: str          # policy_id
    page: int
    type: str         # "parent" | "child"
    parent_id: str = ""
    heading: str = ""
    text: str = ""
    embedding: Optional[List[float]] = None


# ── API request / response models ─────────────────────────────

class SearchRequest(BaseModel):
    question: str
    policy_id: Optional[str] = None
    top_k: int = 5


class SearchResponse(BaseModel):
    question: str
    answer: str
    policy_id: Optional[str] = None
    chunks: List[Dict] = []
    metadata: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    policies_indexed: int = 0
    chunks_indexed: int = 0
