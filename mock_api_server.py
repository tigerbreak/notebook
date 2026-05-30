"""
Mock API 服务器 — 模拟公司内部 API 服务
提供几个简单的接口供 Agent 调用演示
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time
from typing import Optional

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
