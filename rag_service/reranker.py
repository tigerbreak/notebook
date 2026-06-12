"""Cross-encoder Reranker using BAAI/bge-reranker-v2-m3."""

import logging
from typing import List, Tuple, Optional

from config import RERANKER_MODEL, RERANKER_DEVICE, RERANK_TOP_K

log = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for re-scoring retrieval results."""

    def __init__(self, model_name: str = RERANKER_MODEL, device: str = RERANKER_DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self._ready = False

    def load(self) -> bool:
        """Load the cross-encoder model. Returns True if successful."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            log.info(f"🔄 加载 Reranker 模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype="auto",
            )
            self.model.eval()
            self._ready = True
            log.info(f"✅ Reranker 加载完成 ({self.model_name})")
            return True
        except Exception as e:
            log.warning(f"⚠️ Reranker 加载失败（将跳过精排）: {e}")
            self._ready = False
            return False

    def rerank(
        self,
        query: str,
        chunks: List[Tuple],
        top_k: int = RERANK_TOP_K,
    ) -> List[Tuple]:
        """
        Re-score chunks using cross-encoder.

        Args:
            query: User query string.
            chunks: List of (chunk, score, source) tuples.
            top_k: Number of candidates to rerank.

        Returns:
            Re-ranked list of (chunk, new_score, source) tuples, limited to FINAL_TOP_K.
        """
        if not self._ready or not chunks:
            return chunks

        # Take top candidates
        candidates = chunks[:top_k]
        texts = [c[0].text for c in candidates]

        try:
            import torch

            pairs = [[query, text] for text in texts]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).tolist()

            if isinstance(scores, float):
                scores = [scores]

            # Re-sort by reranker score
            scored = []
            for (ch, orig_score, src), new_score in zip(candidates, scores):
                scored.append((ch, float(new_score), src))

            scored.sort(key=lambda x: -x[1])
            return scored

        except Exception as e:
            log.warning(f"⚠️ Reranker 推理失败，使用原始排序: {e}")
            return chunks

    @property
    def is_ready(self) -> bool:
        return self._ready


# Global singleton
reranker = Reranker()
