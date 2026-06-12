#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────
# VM 一键部署脚本 — 保险保单受益人变更检索系统
# 适用于 42.194.162.251 (Tencent Cloud Ubuntu 24.04)
# ───────────────────────────────────────────────────────────
set -euo pipefail

echo "=== 1. 安装 Docker & Docker Compose ==="
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | bash
fi
if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
  pip3 install docker-compose 2>/dev/null || true
fi
DOCKER_COMPOSE="docker-compose"
docker compose version &>/dev/null && DOCKER_COMPOSE="docker compose"

echo "=== 2. 安装系统依赖 ==="
apt-get update -qq && apt-get install -y -qq python3-pip jq 2>/dev/null

echo "=== 3. 克隆代码 ==="
cd /root
[ -d notebook ] || git clone https://github.com/tigerbreak/notebook.git
cd notebook && git pull

echo "=== 4. 配置环境变量 ==="
cd rag_service
if [ ! -f .env ]; then
  echo "⚠️  请设置 DeepSeek API Key:"
  read -s -p "输入 API Key: " api_key
  echo ""
  cat > .env <<EOF
DEEPSEEK_API_KEY=${api_key}
DEEPSEEK_BASE_URL=https://api.deepseek.com
EOF
  echo "✅ .env 已创建"
fi

echo "=== 5. 安装 Elasticsearch IK 分词插件 ==="
# ES will auto-install IK via the init script in docker-compose
# For now, we pull the plugin manually into a volume
ES_CONTAINER="rag-es"
if docker ps --format '{{.Names}}' | grep -q "^${ES_CONTAINER}$"; then
  echo "IK 插件已安装，跳过"
else
  echo "IK 插件将在 ES 首次启动时安装 (docker-compose up 时自动执行)"
fi

echo "=== 6. 启动所有服务 (Docker Compose) ==="
${DOCKER_COMPOSE} down 2>/dev/null || true
${DOCKER_COMPOSE} up -d --build

echo ""
echo "等待服务启动..."
sleep 10

echo "=== 7. 健康检查 ==="
for i in $(seq 1 12); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/health 2>/dev/null || echo "000")
  if [ "$STATUS" = "200" ]; then
    echo "✅ API 服务就绪"
    curl -s http://localhost:8000/api/v1/health | jq .
    break
  fi
  echo "  等待中... ($i/12)"
  sleep 5
done

echo ""
echo "✅ 部署完成！"
echo "   API 地址:    http://localhost:8000/api/v1"
echo "   健康检查:    curl http://localhost:8000/api/v1/health"
echo "   检索测试:    curl -X POST http://localhost:8000/api/v1/search \\"
echo "                 -H 'Content-Type: application/json' \\"
echo "                 -d '{\"question\":\"张美玲受益人变更\",\"policy_id\":\"P0242025-1883\"}'"
echo "   前端界面:    http://localhost:8501"
echo "   日志查看:    ${DOCKER_COMPOSE} logs -f api"
