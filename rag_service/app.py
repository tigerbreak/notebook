"""FastAPI service for Insurance Policy Beneficiary Change Retrieval."""

import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, PORT, API_PREFIX
from models import (
    SearchRequest, SearchResponse, HealthResponse,
    UploadResponse, PolicyDetailResponse, ChunkItem,
)
from engine import engine
from llm import answer as llm_answer, check_connection as check_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 正在构建检索索引…")
    engine.build()
    ctx = f"{len(engine.chunks)} chunks, {len(engine.policies)} policies"
    log.info(f"✅ 索引构建完成：{ctx}")
    log.info(f"   向量: {engine.active_db}  |  全文: {engine.active_fts}  |  精排: {engine.reranker_status}")
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

# ── Pre-warm model cache on import ──
# This ensures the model is loaded at import time, not during lifespan,
# which avoids multi-process hangs with sentence-transformers on Windows.
_log = logging.getLogger(__name__)
try:
    from engine import _get_embedder
    _get_embedder()
    _log.info("✅ Embedding model cache warmed")
except Exception:
    _log.warning("⚠️ Embedding model warm-up skipped")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════

@app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model="deepseek-v4-flash + all-MiniLM-L6-v2",
        policies_indexed=len(engine.policies),
        chunks_indexed=len(engine.chunks),
        vector_db=engine.active_db,
        fulltext_db=engine.active_fts,
        reranker=engine.reranker_status,
    )


# ── Backward-compatible alias ──
@app.get("/health", include_in_schema=False)
async def health_legacy():
    return await health()


# ═══════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════

@app.post(f"{API_PREFIX}/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not engine.is_ready:
        raise HTTPException(503, "索引未就绪")
    if not req.question.strip():
        raise HTTPException(422, "question 不能为空")

    chunks = engine.hybrid_search(req.question, req.policy_id, req.top_k)
    context = engine.get_context(chunks)
    log.info(f"检索结果: {len(chunks)} chunks, context={len(context)} chars")

    pid = req.policy_id or (chunks[0][0].pid if chunks else None)
    metadata = engine.get_metadata_str(pid) if pid else ""
    answer_text = llm_answer(context, metadata, req.question)

    chunk_list = [
        ChunkItem(
            cid=c.cid, policy_id=c.pid, page=c.page,
            heading=c.heading, text=c.text[:200], score=round(s, 4), source=src,
        )
        for c, s, src in chunks
    ]
    return SearchResponse(
        question=req.question,
        answer=answer_text,
        policy_id=pid,
        chunks=chunk_list,
        metadata={
            "policy_id": pid,
            "policies_indexed": len(engine.policies),
            "chunks_indexed": len(engine.chunks),
            "vector_db": engine.active_db,
            "fulltext_db": engine.active_fts,
            "reranker": engine.reranker_status,
        },
    )


@app.post(f"{API_PREFIX}/answer")
async def answer_only(req: SearchRequest):
    """Retrieve + generate without returning chunks."""
    resp = await search(req)
    return {"question": resp.question, "answer": resp.answer}


# ── Backward-compatible aliases ──
@app.post("/search", include_in_schema=False)
async def search_legacy(req: SearchRequest):
    return await search(req)

@app.post("/answer", include_in_schema=False)
async def answer_only_legacy(req: SearchRequest):
    return await answer_only(req)


# ═══════════════════════════════════════════════════════════════
# Policy Upload (async via BackgroundTasks)
# ═══════════════════════════════════════════════════════════════

# In-memory task status store (PoC; production would use Redis)
_tasks: dict = {}


async def _process_upload(task_id: str, policy_data: dict):
    """Simulate async TIF → OCR → chunk → index pipeline."""
    try:
        _tasks[task_id] = {"status": "processing", "message": "正在解析保单..."}
        # Step 1: metadata extraction
        from engine import extract_metadata, chunk_policy
        meta = extract_metadata(policy_data)
        pid = meta.id or policy_data.get("id", "unknown")
        _tasks[task_id] = {"status": "processing", "message": f"正在分块 ({pid})..."}

        # Step 2: chunking
        chunks = chunk_policy(policy_data)
        _tasks[task_id] = {"status": "processing", "message": f"正在索引 ({len(chunks)} chunks)..."}

        # Step 3: embedding & indexing
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        child_chunks = [c for c in chunks if c.type == "child"]
        if child_chunks:
            embeddings = embedder.encode([c.text for c in child_chunks], show_progress_bar=False)
            if engine.qdrant.is_ready:
                engine.qdrant.add_chunks(child_chunks, embeddings.tolist())
            if engine.es.is_ready:
                engine.es.add_chunks(child_chunks)

        # Update engine state
        engine.policies.append(policy_data)
        engine.metas[pid] = meta
        engine.chunks.extend(chunks)
        engine.children.extend(child_chunks)

        _tasks[task_id] = {
            "status": "completed",
            "message": f"保单 {pid} 处理完成，共 {len(chunks)} 个分块",
            "policy_id": pid,
        }
    except Exception as e:
        log.exception("上传处理失败")
        _tasks[task_id] = {"status": "failed", "message": f"处理失败: {e}"}


@app.post(f"{API_PREFIX}/policies/upload", response_model=UploadResponse)
async def upload_policy(background_tasks: BackgroundTasks):
    """Upload a new policy (mock: uses built-in sample data)."""
    if not engine.is_ready:
        raise HTTPException(503, "索引未就绪")

    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {"status": "queued", "message": "任务已加入队列"}

    # In PoC, process the next mock policy not yet indexed
    indexed_ids = set(engine.metas.keys())
    for pol in engine.policies:
        if pol["id"] not in indexed_ids:
            background_tasks.add_task(_process_upload, task_id, pol)
            return UploadResponse(
                task_id=task_id,
                status="queued",
                message=f"正在处理保单 {pol['id']}",
            )

    return UploadResponse(
        task_id=task_id,
        status="completed",
        message="所有保单已索引，无新保单可上传",
    )


@app.get(f"{API_PREFIX}/tasks/{{task_id}}")
async def get_task_status(task_id: str):
    """Poll task status."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, f"任务 {task_id} 不存在")
    return task


# ═══════════════════════════════════════════════════════════════
# Policy Detail
# ═══════════════════════════════════════════════════════════════

@app.get(f"{API_PREFIX}/policies/{{policy_id}}", response_model=PolicyDetailResponse)
async def get_policy(policy_id: str):
    """Get policy metadata and chunk overview."""
    detail = engine.get_policy_detail(policy_id)
    if not detail:
        raise HTTPException(404, f"保单 {policy_id} 不存在")

    chunks = [c for c in engine.chunks if c.pid == policy_id and c.type == "child"]
    chunk_items = [
        ChunkItem(
            cid=c.cid, policy_id=c.pid, page=c.page,
            heading=c.heading, text=c.text[:200], score=0.0,
        )
        for c in chunks
    ]
    return PolicyDetailResponse(
        policy_id=policy_id,
        metadata=detail["metadata"],
        chunks=chunk_items,
        page_count=detail["page_count"],
    )


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
