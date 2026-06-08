"""
Mock API 服务器 — 模拟公司内部 API 服务
提供几个简单的接口供 Agent 调用演示
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time
from typing import Any, Optional

app = FastAPI(title="Mock Company API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API 根路径"""
    return {"message": "Welcome to Mock Company API", "version": "1.0.0"}


@app.get("/api/employees")
async def get_employees(department: Optional[str] = None):
    """获取员工列表"""
    employees = [
        {"id": 1, "name": "张三", "department": "工程部", "position": "高级工程师"},
        {"id": 2, "name": "李四", "department": "产品部", "position": "产品经理"},
        {"id": 3, "name": "王五", "department": "工程部", "position": "架构师"},
        {"id": 4, "name": "赵六", "department": "市场部", "position": "市场总监"},
        {"id": 5, "name": "钱七", "department": "人事部", "position": "HR 主管"},
    ]
    if department:
        employees = [e for e in employees if e["department"] == department]
    return {"total": len(employees), "employees": employees}


@app.get("/api/projects")
async def get_projects(status: Optional[str] = None):
    """获取项目列表"""
    projects = [
        {"id": "P001", "name": "AI 智能客服系统", "status": "进行中", "progress": 75, "team": "工程部"},
        {"id": "P002", "name": "数据中台重构", "status": "进行中", "progress": 40, "team": "工程部"},
        {"id": "P003", "name": "Q4 市场推广计划", "status": "规划中", "progress": 10, "team": "市场部"},
        {"id": "P004", "name": "员工培训平台", "status": "已完成", "progress": 100, "team": "人事部"},
    ]
    if status:
        projects = [p for p in projects if p["status"] == status]
    return {"total": len(projects), "projects": projects}


@app.get("/api/metrics")
async def get_metrics():
    """获取公司关键指标"""
    await asyncio.sleep(0.5)  # 模拟 API 延迟
    return {
        "revenue": {"q3": 12500000, "q4_target": 15000000, "unit": "CNY"},
        "users": {"active": 50000, "new_this_month": 3200},
        "satisfaction": 4.6,
        "uptime": 99.95,
    }


@app.post("/api/search")
async def search_company(query: str):
    """内部搜索接口"""
    await asyncio.sleep(0.3)
    results = {
        "ai": [
            "公司正在推进 AI 智能客服系统项目 (P001)，已完成 75%",
            "工程部有两位高级工程师负责 AI 相关开发",
        ],
        "员工": [
            "公司共有 5 名员工分布在 4 个部门",
            "工程部有 2 名员工：张三（高级工程师）和王五（架构师）",
        ],
        "项目": [
            "目前有 4 个项目，其中 2 个进行中，1 个规划中，1 个已完成",
        ],
    }
    matched = []
    for key, items in results.items():
        if key in query.lower() or key in query:
            matched.extend(items)
    if not matched:
        matched = [f"找到 {len(query)} 个字符的搜索查询结果", f"未找到与 '{query}' 完全匹配的内容"]
    return {"query": query, "results": matched}


# ============================================================
# RAG 向量检索端点 — 模拟企业知识库检索
# ============================================================

DOCUMENT_CHUNKS = [
    {
        "content": "公司 AI 战略白皮书：2024 年 AI 投入预算 5000 万元，重点布局大语言模型、智能客服和数据分析三个方向。",
        "source": "AI 战略规划_v2.1.pdf",
        "page": 3,
    },
    {
        "content": "AI 智能客服系统项目 (P001)：由工程部负责，目标是将客服响应时间从平均 5 分钟缩短到 30 秒以内，预计节省人力成本 200 万元/年。",
        "source": "项目立项报告_P001.pdf",
        "page": 1,
    },
    {
        "content": "数据中台项目 (P002)：当前进度 40%，已完成数据仓库架构设计和核心 ETL 管道搭建，预计 Q4 完成数据湖建设。",
        "source": "数据中台季度报告_Q3.pdf",
        "page": 5,
    },
    {
        "content": "公司技术栈选型指南：后端推荐使用 Python + FastAPI，前端推荐 React + TypeScript，数据库推荐 PostgreSQL + Redis 缓存。",
        "source": "技术委员会决议_2024.pdf",
        "page": 12,
    },
    {
        "content": "员工培训体系：2024 年已完成 3 期技术培训，覆盖 120 人次，包括 AI/ML 入门、云原生架构、微服务设计模式等主题。",
        "source": "培训年度报告_2024.pdf",
        "page": 8,
    },
    {
        "content": "财务部 Q3 财报：营收 1250 万元，同比增长 18%，净利润率 15.2%。Q4 目标营收 1500 万元。",
        "source": "季度财报_Q3_2024.pdf",
        "page": 2,
    },
    {
        "content": "产品路线图 2024H2：旗舰产品 v3.0 将于 11 月发布，新增 AI 推荐引擎、多语言支持和实时协作功能。",
        "source": "产品规划_2024H2.pdf",
        "page": 1,
    },
    {
        "content": "安全合规报告：公司已通过 ISO27001 认证，数据安全等级保护三级，每年进行两次渗透测试和代码安全审计。",
        "source": "安全合规年报_2024.pdf",
        "page": 15,
    },
    {
        "content": "市场部 Q4 推广计划：预算 300 万元，目标新增用户 5 万，重点渠道为社交媒体和内容营销。",
        "source": "市场推广计划_Q4.pdf",
        "page": 3,
    },
    {
        "content": "研发效能指标：2024 年代码提交量同比增长 25%，平均代码评审周期 1.5 天，CI/CD 流水线通过率 98%。",
        "source": "研发效能报告_Q3.pdf",
        "page": 7,
    },
]


def _compute_similarity(query: str, content: str) -> float:
    """简单的关键词匹配相似度计算（模拟向量检索）"""
    query_words = set(query.lower().replace("的", "").replace("和", "").replace("关于", "").split())
    content_words = set(content.lower())
    if not query_words:
        return 0.0
    overlap = len(query_words & content_words)
    return min(overlap / max(len(query_words), 1) + 0.3, 1.0)


@app.post("/api/rag/search")
async def rag_search(request: dict):
    """
    RAG 向量检索接口 — 模拟企业知识库检索
    参照 RAGFlow chunk 格式返回结果
    """
    await asyncio.sleep(0.5)  # 模拟检索延迟
    
    query = request.get("query", "")
    top_k = request.get("top_k", 5)
    min_score = request.get("min_score", 0.5)
    include_citations = request.get("include_citations", True)
    
    # 计算每条文档的相似度分数
    scored_chunks = []
    for chunk in DOCUMENT_CHUNKS:
        score = _compute_similarity(query, chunk["content"])
        scored_chunks.append({
            "content": chunk["content"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": round(score, 4)
        })
    
    # 按分数排序，过滤低分结果
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    results = [c for c in scored_chunks if c["score"] >= min_score][:top_k]
    
    if not include_citations:
        for r in results:
            r.pop("source", None)
            r.pop("page", None)
    
    return {"query": query, "total": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
