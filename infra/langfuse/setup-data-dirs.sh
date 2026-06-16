#!/usr/bin/env bash
# 创建 Langfuse 各服务的数据目录并设置正确的 owner
# 各容器内运行用户的 UID/GID：
#   postgres (alpine)   : 999:999
#   clickhouse (alpine) : 101:101
#   redis (alpine)      : 999:1000  （实际无强制要求，root 也能写）
#   minio               : 1000:1000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 创建 data 子目录 ==="
mkdir -p data/postgres data/clickhouse data/redis data/minio

echo ""
echo "=== 设置 owner（让容器内用户能写入）==="
sudo chown -R 999:999   data/postgres
sudo chown -R 101:101   data/clickhouse
sudo chown -R 1000:1000 data/minio
# redis alpine 镜像默认以 root 跑，权限无所谓
sudo chown -R 1000:1000 data/redis

echo ""
echo "=== 完成 ==="
ls -la data/
echo ""
echo "下一步：编辑 .env，然后 docker compose up -d"
