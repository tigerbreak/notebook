"""
RAG 检索增强生成工具
参照 RAGFlow agent/tools/tavily.py 的 ToolParamBase + ToolBase 分离设计
"""

from typing import Any, Dict, List, Optional

from .base import BaseTool


class RAGSearch(BaseTool):
    """
    RAG 向量检索工具 — 在企业知识库中检索相关文档片段。
    
    参照 RAGFlow TavilySearch 模式：
    - 参数：query (必需), top_k, min_score, include_citations
    - 支持重试和错误处理
    - 返回格式化的检索结果供 LLM 使用
    """
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8080",
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "rag_search"
    
    @property
    def description(self) -> str:
        return (
            "在企业知识库中进行 RAG 向量检索。适用于需要查找文档片段、"
            "技术规范、历史文档等场景。返回带引用来源和置信度分数的检索结果。"
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询语句，建议简短具体"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回最相关的 top_k 个结果，默认 5",
                    "default": 5
                },
                "min_score": {
                    "type": "number",
                    "description": "最低相似度分数阈值 (0-1)，默认 0.5",
                    "default": 0.5
                },
                "include_citations": {
                    "type": "boolean",
                    "description": "是否包含引用来源信息，默认 true",
                    "default": True
                }
            },
            "required": ["query"]
        }
    
    async def __call__(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.5,
        include_citations: bool = True,
        **kwargs
    ) -> str:
        import httpx
        
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.api_base_url}/api/rag/search",
                        json={
                            "query": query,
                            "top_k": top_k,
                            "min_score": min_score,
                            "include_citations": include_citations
                        }
                    )
                    
                    if response.status_code != 200:
                        return f"RAG API error: {response.status_code} - {response.text}"
                    
                    data = response.json()
                    return self._format_results(data.get("results", []), include_citations)
                    
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    continue
                return f"RAG search timeout after {self.max_retries} attempts"
            except Exception as e:
                return f"RAG search error: {str(e)}"
        
        return "RAG search failed after all retries"
    
    def _format_results(self, results: List[Dict], include_citations: bool = True) -> str:
        """将结构化 RAG 结果转换为 LLM 友好的文本格式"""
        if not results:
            return "未找到相关的知识库检索结果。"
        
        formatted = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            score = result.get("score", 0.0)
            
            entry = f"[{i}] (score: {score:.3f}) {content}"
            
            if include_citations:
                source = result.get("source", "未知来源")
                page = result.get("page", "")
                citation = f"\n    📎 来源: {source}"
                if page:
                    citation += f" (第 {page} 页)"
                entry += citation
            
            formatted.append(entry)
        
        return "\n\n".join(formatted)
