"""Streamlit frontend for Insurance Policy Beneficiary Change Retrieval."""

import os
import streamlit as st
import requests

# ── Config ────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = os.environ.get("API_PORT", "8000")
API_BASE = f"http://{API_HOST}:{API_PORT}/api/v1"

st.set_page_config(
    page_title="保险保单受益人变更检索系统",
    page_icon="📋",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/insurance--v1.png", width=64)
    st.title("保单检索系统")
    st.caption("RAG-based · DeepSeek + Hybrid Search")

    st.divider()
    st.subheader("系统状态")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if r.ok:
            data = r.json()
            st.success(f"✅ 服务正常")
            st.metric("已索引保单", data.get("policies_indexed", 0))
            st.metric("已索引分块", data.get("chunks_indexed", 0))
            st.caption(f"向量: {data.get('vector_db', '?')}")
            st.caption(f"全文: {data.get('fulltext_db', '?')}")
            st.caption(f"精排: {data.get('reranker', '?')}")
        else:
            st.error("❌ 服务异常")
    except Exception as e:
        st.error(f"❌ 无法连接 API: {e}")

    st.divider()
    st.caption("保险保单受益人变更检索系统 v1.0")

# ── Main ──────────────────────────────────────────────────────
st.title("🔍 保单受益人变更检索")

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input(
        "输入查询问题",
        placeholder="例如：张美玲的受益比例是多少？",
        label_visibility="collapsed",
    )
with col2:
    policy_id = st.text_input(
        "保单号（可选过滤）",
        placeholder="留空表示全部",
        label_visibility="collapsed",
    )

search_btn = st.button("🚀 检索", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────
if search_btn and question.strip():
    with st.spinner("正在检索并生成答案..."):
        try:
            resp = requests.post(
                f"{API_BASE}/search",
                json={"question": question, "policy_id": policy_id or None, "top_k": 5},
                timeout=60,
            )
            if not resp.ok:
                st.error(f"API 错误: {resp.status_code} - {resp.text}")
                st.stop()

            data = resp.json()
        except Exception as e:
            st.error(f"请求失败: {e}")
            st.stop()

    # Display answer
    st.subheader("💡 答案")
    st.markdown(data["answer"])

    # Display chunks
    st.divider()
    st.subheader(f"📄 参考片段 ({len(data['chunks'])} 个)")

    for i, chunk in enumerate(data["chunks"], 1):
        with st.expander(f"#{i} 保单 {chunk['policy_id']} · 第 {chunk['page']} 页 · 得分 {chunk['score']:.4f}"):
            st.caption(f"段落: {chunk.get('heading', '(正文)')}   |   来源: {chunk.get('source', 'hybrid')}")
            st.text(chunk["text"])

    # Metadata
    if data.get("metadata"):
        with st.expander("📊 元数据"):
            st.json(data["metadata"])

elif search_btn and not question.strip():
    st.warning("请输入查询问题")

# ── Example queries ───────────────────────────────────────────
st.divider()
with st.expander("💡 示例查询"):
    examples = [
        "张美玲的受益比例是多少？",
        "王建国的保单有哪些受益人变更？",
        "查询保单 P0242025-1883 的受益人信息",
        "TPK-2023-004517 发生过几次受益人变更？",
        "PCI-2024-7721 是否有受益人变更记录？",
        "批单 BG2024-00137 的变更内容是什么？",
    ]
    for ex in examples:
        if st.button(f"📝 {ex}", use_container_width=True):
            st.session_state["example_query"] = ex
            st.rerun()

# Auto-fill from example click
if "example_query" in st.session_state:
    question = st.session_state.pop("example_query")
    st.rerun()
