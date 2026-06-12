"""API test script for the RAG service."""
import urllib.request, json, sys

BASE = "http://localhost:8000/api/v1"

def test_health():
    print("=" * 50)
    print("1️⃣  Health Check")
    resp = urllib.request.urlopen(f"{BASE}/health", timeout=10)
    data = json.loads(resp.read())
    print(f"   Status: {data['status']}")
    print(f"   Policies: {data['policies_indexed']}")
    print(f"   Chunks: {data['chunks_indexed']}")
    print(f"   Vector DB: {data['vector_db']}")
    print(f"   Fulltext DB: {data['fulltext_db']}")
    print(f"   Reranker: {data['reranker']}")
    assert data['status'] == 'ok'
    assert data['policies_indexed'] == 3
    print("   ✅ Health check passed")
    return data

def test_search(question, policy_id=None):
    print(f"\n{'=' * 50}")
    print(f"2️⃣  Search: {question}")
    body = json.dumps({"question": question, "policy_id": policy_id}).encode()
    req = urllib.request.Request(f"{BASE}/search", data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    
    print(f"   Answer: {data['answer'][:300]}...")
    print(f"   Chunks: {len(data['chunks'])}")
    for i, c in enumerate(data['chunks'], 1):
        src = c.get('source', 'hybrid')
        print(f"     #{i} 保单:{c['policy_id']} 第{c['page']}页 score={c['score']:.4f} src={src}")
        print(f"        {c['text'][:80]}...")
    assert len(data['chunks']) > 0
    print("   ✅ Search passed")
    return data

def test_policy_detail(policy_id):
    print(f"\n{'=' * 50}")
    print(f"3️⃣  Policy Detail: {policy_id}")
    resp = urllib.request.urlopen(f"{BASE}/policies/{policy_id}", timeout=10)
    data = json.loads(resp.read())
    print(f"   Policy: {data['policy_id']}")
    print(f"   Pages: {data['page_count']}")
    print(f"   Chunks: {len(data['chunks'])}")
    meta = data.get('metadata', {})
    print(f"   Insurer: {meta.get('insurer', 'N/A')}")
    print(f"   Applicant: {meta.get('applicant', 'N/A')}")
    assert data['policy_id'] == policy_id
    print("   ✅ Policy detail passed")
    return data

def test_answer_only(question):
    print(f"\n{'=' * 50}")
    print(f"4️⃣  Answer Only: {question}")
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(f"{BASE}/answer", data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    print(f"   Answer: {data['answer'][:200]}...")
    print("   ✅ Answer-only passed")
    return data

def test_legacy_endpoints():
    print(f"\n{'=' * 50}")
    print("5️⃣  Legacy Backward Compat")
    # Legacy /health
    resp = urllib.request.urlopen("http://localhost:8000/health", timeout=10)
    data = json.loads(resp.read())
    assert data['status'] == 'ok'
    # Legacy /search
    body = json.dumps({"question": "test"}).encode()
    req = urllib.request.Request("http://localhost:8000/search", data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
    assert len(data['chunks']) > 0
    print("   ✅ Legacy endpoints passed")
    return data

if __name__ == "__main__":
    print("🚀 开始 API 闭环测试\n")
    try:
        test_health()
        test_search("张美玲的受益比例是多少？", "P0242025-1883")
        test_search("TPK-2023-004517 发生过几次受益人变更？")
        test_policy_detail("P0242025-1883")
        test_answer_only("王建国的保单有哪些受益人？")
        test_legacy_endpoints()
        print(f"\n{'=' * 50}")
        print("✅ 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
