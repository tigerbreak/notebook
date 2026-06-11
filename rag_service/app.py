"""FastAPI service for Insurance Policy Beneficiary Change Retrieval."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, PORT
from models import SearchRequest, SearchResponse, HealthResponse
from engine import engine
from llm import answer as llm_answer, check_connection as check_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 正在构建检索索引…")
    engine.build()
    log.info(f"✅ 索引构建完成：{len(engine.chunks)} chunks, {len(engine.policies)} policies")
    llm_ok = check_llm()
    log.info(f"{'✅' if llm_ok else '❌'} DeepSeek API 连接: {'正常' if llm_ok else '失败'}")
    yield
    log.info("👋 服务关闭")


app = FastAPI(
    title="保险保单受益人变更检索系统",
    description="RAG-based retrieval for insurance policy beneficiary changes",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model="deepseek-v4-flash + all-MiniLM-L6-v2",
        policies_indexed=len(engine.policies),
        chunks_indexed=len(engine.chunks),
    )


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not engine.is_ready:
        raise HTTPException(503, "Index not ready")
    if not req.question.strip():
        raise HTTPException(422, "question is required")

    # 1. Hybrid retrieval
    chunks = engine.hybrid_search(req.question, req.policy_id, req.top_k)

    # 2. Build context
    context = engine.get_context(chunks)
    log.info(f"检索结果: {len(chunks)} chunks, context={len(context)} chars")

    # 3. Metadata
    pid = req.policy_id or (chunks[0][0].pid if chunks else None)
    metadata = engine.get_metadata_str(pid) if pid else ""

    # 4. LLM answer
    answer_text = llm_answer(context, metadata, req.question)

    # 5. Response
    chunk_list = [
        {"cid": c.cid, "policy_id": c.pid, "page": c.page,
         "heading": c.heading, "text": c.text[:200], "score": round(s, 4)}
        for c, s, src in chunks
    ]
    return SearchResponse(
        question=req.question,
        answer=answer_text,
        policy_id=pid,
        chunks=chunk_list,
        metadata={"policy_id": pid, "policies_indexed": len(engine.policies),
                  "chunks_indexed": len(engine.chunks)},
    )


@app.post("/answer")
async def answer_only(req: SearchRequest):
    """Retrieve + generate without returning chunks."""
    resp = await search(req)
    return {"question": resp.question, "answer": resp.answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
