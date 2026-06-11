#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────
# VM 一键部署脚本 — 保险保单受益人变更检索系统
# 适用于 42.194.162.251 (Tencent Cloud Ubuntu 24.04)
# ───────────────────────────────────────────────────────────
set -euo pipefail

echo "=== 1. 安装系统依赖 ==="
apt-get update -qq && apt-get install -y -qq python3-pip jq 2>/dev/null
pip3 install --break-system-packages --no-deps -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com \
  torch==2.6.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null

echo "=== 2. 安装 Python 包 ==="
pip3 install --break-system-packages --ignore-installed -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com \
  fastapi uvicorn openai pydantic python-multipart jieba rank-bm25 \
  sentence-transformers 2>&1 | tail -3

pip3 install --break-system-packages --no-deps -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com \
  transformers==4.48.3 2>&1 | tail -1

pip3 install --break-system-packages -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com \
  'tokenizers>=0.21,<0.22' 2>&1 | tail -1

echo "=== 3. 验证安装 ==="
python3 -c "
import torch; print(f'torch {torch.__version__} ({(\"CPU\" if not torch.cuda.is_available() else \"CUDA\")})')
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print(f'embedding dim={m.get_sentence_embedding_dimension()}')
"

echo "=== 4. 克隆代码 ==="
cd /root
[ -d notebook ] || git clone https://github.com/tigerbreak/notebook.git
cd notebook && git pull && git checkout architecture-overview

echo "=== 5. 启动服务 ==="
cd rag_service
pkill -f "uvicorn app:app" 2>/dev/null || true
echo "⚠️  请先设置 DeepSeek API Key: export DEEPSEEK_API_KEY='your-key'"
nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/rag_service.log 2>&1 &
sleep 3

echo "=== 6. 健康检查 ==="
curl -s http://localhost:8000/health | jq .
echo ""
echo "✅ 部署完成！API 地址: http://0.0.0.0:8000"
echo "   健康检查: curl http://localhost:8000/health"
echo "   检索测试: curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -d '{\"question\":\"张美玲受益人变更\",\"policy_id\":\"P0242025-1883\"}'"
