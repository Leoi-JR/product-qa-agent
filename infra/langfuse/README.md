# Langfuse 自托管部署

本目录用于在**宿主机**上启动一套完整的 Langfuse v3 服务（6 个容器），给项目提供本地化的 LLM 可观测性（trace、span、评估）。

由于开发容器与宿主机共享项目目录（bind mount），在容器内修改本目录的文件会实时同步到宿主机；启动命令则在宿主机执行。

## 架构

```
开发容器（应用代码）              宿主机（6 个容器）
─────────────────                ─────────────────────────────────
FastAPI                          ┌─ langfuse-web       :3001  ←─ 浏览器 + SDK
  └─ Langfuse SDK                │  ↳ UI + API 入口（唯一暴露端口）
       │  HTTP POST              │
       │  :3001/api/public       ├─ langfuse-worker   （异步处理 trace）
       │                         │  ↳ 消费 Redis 队列，写入 ClickHouse
       └─────────────────────→  │
                                 ├─ langfuse-db        （Postgres 15）
                                 │  ↳ 元数据：用户/项目/API key
                                 │
                                 ├─ langfuse-clickhouse（ClickHouse 24.1）
                                 │  ↳ trace 数据：span/observation
                                 │
                                 ├─ langfuse-redis     （Redis 7）
                                 │  ↳ BullMQ 任务队列
                                 │
                                 └─ langfuse-minio     （MinIO）
                                    ↳ S3 兼容对象存储（trace event/media）
```

Langfuse v3 的服务分层：
- **数据层**：Postgres（元数据）+ ClickHouse（trace 数据）+ Redis（队列）+ MinIO（对象存储）
- **应用层**：langfuse-web（API/UI 入口）+ langfuse-worker（异步处理）

## 端口与冲突规避

| 服务 | 端口策略 |
|------|---------|
| langfuse-db (Postgres) | **不暴露到宿主机**，避开宿主机已有的 Postgres 5432 |
| langfuse-clickhouse | **不暴露到宿主机**，避开宿主机已有的 ClickHouse 8123/9000 |
| langfuse-redis | **不暴露到宿主机**，避开宿主机已有的 Redis 6379 |
| langfuse-minio | **不暴露到宿主机**，避开宿主机已有的 MinIO 9000/9001 |
| langfuse-worker | **不暴露到宿主机**（无外部访问需求） |
| langfuse-web | 映射到宿主机 `3001`（避开常见的 3000），可通过 `LANGFUSE_PORT` 调整 |

## 部署步骤

### 1. 准备数据目录（首次部署）

在**宿主机**执行（路径里的空格用引号包起来）：

```bash
cd "<宿主机映射路径>/商品知识问答 Agent/infra/langfuse"
chmod +x setup-data-dirs.sh
./setup-data-dirs.sh
```

脚本会创建 4 个 data 子目录并设置正确的 owner（容器内用户 UID）：
- `data/postgres/` → `999:999`
- `data/clickhouse/` → `101:101`
- `data/minio/` → `1000:1000`
- `data/redis/` → `1000:1000`

### 2. 生成 .env

```bash
cp .env.example .env
```

**一键生成所有密钥**（在宿主机执行，把输出粘到 `.env`）：

```bash
echo "── NextAuth / SALT / ENCRYPTION_KEY（用 hex 32 字节）──"
for k in NEXTAUTH_SECRET SALT ENCRYPTION_KEY; do
  echo "$k=$(openssl rand -hex 32)"
done

echo "── 数据库密码（base64，去掉 URL 不安全字符）──"
gen() { openssl rand -base64 24 | tr -d '/+=' | head -c 32; }
echo "POSTGRES_PASSWORD=$(gen)"
echo "CLICKHOUSE_PASSWORD=$(gen)"
echo "REDIS_AUTH=$(gen)"
echo "MINIO_ROOT_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 16)"
```

或者手动改 `.env`，所有标 `# CHANGEME` 的字段都要替换。

### 3. 启动服务

```bash
docker compose up -d

# 看启动进度（v3 启动慢，等 1-2 分钟）
docker compose logs -f langfuse-web
```

### 4. 验证启动成功

```bash
# 容器状态
docker compose ps
# 期望：6 个容器都是 Up (healthy)

# Web 健康检查
curl http://localhost:3001/api/public/health
# 期望：{"status":"OK"}

# Worker 状态
docker compose logs --tail 20 langfuse-worker
# 期望：没有报错
```

### 5. 初始化账号与项目

1. 浏览器访问 `http://<宿主机IP>:3001`
2. 注册一个账号（任意邮箱，开发模式已禁用邮件验证）
3. 创建一个 Organization → Project
4. 在 Project Settings → API Keys 拿到 `Public Key` 和 `Secret Key`

### 6. 接入应用

回到开发容器，在项目根 `.env` 追加：

```bash
LANGFUSE_HOST=http://<宿主机IP>:3001
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

`<宿主机IP>` 怎么找（在开发容器内执行）：

```bash
ip route | grep default
# default via 192.168.200.1 dev eth0  ← 网关 IP 就是宿主机
```

写法：
```bash
LANGFUSE_HOST=http://192.168.200.1:3001
```

## 日常运维

```bash
# 查看状态
docker compose ps

# 看日志
docker compose logs -f langfuse-web
docker compose logs -f langfuse-worker

# 停止
docker compose down

# 升级 Langfuse 版本
docker compose pull
docker compose up -d

# 彻底清空（注意：会删所有 trace 数据和已注册账号）
docker compose down
sudo rm -rf data/
./setup-data-dirs.sh   # 重建空目录
```

## 数据存储位置

所有数据用 bind mount，肉眼可见：

| 服务 | 数据目录 | 容器内用户 |
|------|---------|-----------|
| Postgres | `./data/postgres/` | 999:999 |
| ClickHouse | `./data/clickhouse/` | 101:101 |
| Redis | `./data/redis/` | 1000:1000 |
| MinIO | `./data/minio/` | 1000:1000 |

如需迁移到其他机器：`docker compose down` → 打包整个 `data/` → 在新机器解压 → `docker compose up -d`。

## 故障排查

### 症状：langfuse-web 反复重启

```bash
# 1. 看日志
docker logs langfuse-web --tail 50
docker inspect langfuse-web --format='{{.RestartCount}}'

# 2. 常见原因
# - ClickHouse migration 失败 → 检查 CLICKHOUSE_MIGRATION_URL 端口是 9000 不是 8123
# - "CLICKHOUSE_MIGRATION_URL is not configured" → 检查 .env 是否完整
# - "failed to open database" → 检查 CLICKHOUSE_DB 是 default，密码无特殊字符
# - "SHOW TABLES FROM default" 失败 → 用 setup-data-dirs.sh 重建 data/clickhouse 权限
```

### 症状：浏览器打不开 3001

```bash
docker compose ps           # langfuse-web 是不是 healthy
sudo ss -tlnp | grep 3001   # 端口是否真起来
docker compose logs langfuse-web --tail 30
```

### 症状：开发容器连不上 Langfuse

```bash
# 容器内验证
ip route | grep default    # 拿宿主机 IP
curl http://<网关IP>:3001/api/public/health

# 如果 TCP 通但 HTTP 被 reset
ps aux | grep -iE "v2ray|xray" | grep -v grep
# 透明代理（v2raya）会拦截 HTTP，停掉它们再试
```

### 症状：trace 上报了但 UI 看不到

```bash
# 1. 检查 worker 是否健康
docker compose logs langfuse-worker --tail 50

# 2. 检查 ClickHouse 是否能查到数据
docker exec langfuse-clickhouse clickhouse-client \
    --user langfuse --password <你的密码> \
    --query "SELECT count() FROM default.events"

# 3. 检查 MinIO 是否能写入
docker exec langfuse-minio mc ls local/langfuse/
```

## 文件清单

```
infra/langfuse/
├── docker-compose.yml         # 主配置（6 个服务）
├── .env.example               # 环境变量模板
├── .env                       # 本地实际密钥（不入库）
├── .gitignore                 # 忽略 .env 和 data/
├── setup-data-dirs.sh         # 创建 data 目录并设置权限
├── clickhouse-config.xml      # ClickHouse IPv4-only 监听配置
├── README.md                  # 本文档
└── data/                      # 数据目录（不入库）
    ├── postgres/
    ├── clickhouse/
    ├── redis/
    └── minio/
```

## 选型说明

为什么用这套完整配置而不是精简版：

| 组件 | 必要性 | 说明 |
|------|--------|------|
| Postgres | ✅ 必需 | 元数据存储 |
| ClickHouse | ✅ 必需 | v3 强制依赖，trace 数据 |
| Redis | ✅ 必需 | worker 的 BullMQ 队列 |
| MinIO | ✅ 必需 | v3 强制要求 S3 后端（event upload） |
| Worker | ✅ 必需 | 异步处理 trace 写入 ClickHouse |
| Web | ✅ 必需 | UI + API 入口 |

之前尝试精简（去掉 Redis/Worker/MinIO）导致 v3 启动失败——Langfuse v3 把这些组件视为硬性依赖，无法跳过。
